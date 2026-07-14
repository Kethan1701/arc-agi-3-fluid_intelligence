"""
decision_core.py
=================
`HybridDecisionCore`: the shared hybrid-routing brain, decoupled from
`arcengine`/the Kaggle harness. `agent/my_agent.py` (the one file the
official ARC-AGI-3-Kaggle-Starter contract lets you write) is a thin
adapter around one shared instance of this class.

Why a single shared instance: verified against `agents/swarm.py` in the
official ARC-AGI-3-Agents framework, one `Agent` subclass instance is
created PER GAME, and all of them run concurrently in separate threads
within one process. Loading the model (`VLLMPolicy`) once per game would
be both slow and would blow the 9-hour/VRAM budget. Instead,
`agent/my_agent.py` should construct exactly one `HybridDecisionCore`
(module-level singleton, built lazily behind a lock -- see
`get_shared_core()` below) and have every `MyAgent` instance's per-game
logic go through it. The per-game `GameMemory` is still tracked
separately per `game_id` inside this class, so concurrent games don't
see each other's history.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

from hybrid_lib.fallback_solver import FallbackDecision, FallbackSolver
from hybrid_lib.model_runtime import Policy, PolicyOutput, StubPolicy
from hybrid_lib.resource_manager import ResourceBudget, ResourceManager
from hybrid_lib.state import GameMemory, MultiGrid, SessionMemory

logger = logging.getLogger("arc_agent")


@dataclass
class AgentConfig:
    model_dir: str | None = None          # local path; None => StubPolicy (no GPU)
    use_vllm: bool = False                 # explicit opt-in, see build_policy()
    confidence_threshold: float = 0.55     # below this, route to fallback solver
    max_steps_per_game: int = 80           # matches agents.agent.Agent.MAX_ACTIONS' own default
    resource_budget: ResourceBudget | None = None


def build_policy(config: AgentConfig) -> Policy:
    """
    Construct the primary policy backend. Free function (not inlined into
    `HybridDecisionCore.__init__`) so tests can swap in a fake `Policy`.
    """
    if not config.use_vllm or not config.model_dir:
        logger.info(
            "Using StubPolicy (no GPU model loaded) - set config.use_vllm=True "
            "and config.model_dir=<local path> to enable real inference."
        )
        return StubPolicy()

    from hybrid_lib.model_runtime import VLLMPolicy  # noqa: PLC0415

    return VLLMPolicy(model_dir=config.model_dir)


class HybridDecisionCore:
    """
    Shared, thread-safe hybrid routing brain. One instance per submission
    process; `game_memories` is keyed by `game_id` so concurrently-running
    games each get their own episodic memory while sharing one policy,
    resource manager, and session memory.

    Routing order in `choose()`: resource budget exhausted -> fallback;
    detected action loop -> fallback; primary policy confidence below
    `confidence_threshold` -> fallback; else use the primary policy.
    """

    def __init__(self, config: AgentConfig | None = None) -> None:
        self.config = config or AgentConfig()
        self.policy: Policy = build_policy(self.config)
        self.fallback = FallbackSolver()
        self.resources = ResourceManager(self.config.resource_budget)
        self.session_memory = SessionMemory()

        self._memories_lock = threading.Lock()
        self._game_memories: dict[str, GameMemory] = {}

    # ------------------------------------------------------------------
    # Per-game lifecycle
    # ------------------------------------------------------------------
    def start_game(self, game_id: str) -> GameMemory:
        memory = GameMemory(game_id=game_id)
        with self._memories_lock:
            self._game_memories[game_id] = memory
        self.resources.start_game(game_id)
        logger.info("Starting game %s | budget=%s", game_id, self.resources.summary())
        return memory

    def end_game(self, game_id: str, solved: bool) -> None:
        with self._memories_lock:
            memory = self._game_memories.pop(game_id, None)
        if memory is None:
            return
        self.session_memory.ingest(memory, solved=solved)
        self.resources.end_game(game_id)
        logger.info(
            "Ended game %s solved=%s steps=%d | session_solve_rate=%.2f",
            game_id, solved, memory.step_index, self.session_memory.solve_rate,
        )

    def memory_for(self, game_id: str) -> GameMemory:
        with self._memories_lock:
            memory = self._game_memories.get(game_id)
        if memory is None:
            raise KeyError(f"No active GameMemory for {game_id!r} -- call start_game() first")
        return memory

    # ------------------------------------------------------------------
    # Core decision step
    # ------------------------------------------------------------------
    def choose(
        self,
        game_id: str,
        action_space: list[str],
        complex_actions: frozenset[str] = frozenset(),
    ) -> FallbackDecision:
        """
        Choose the next action for `game_id`. Returns a `FallbackDecision`
        (action name + optional action_args + a reasoning string) whether
        the decision came from the primary policy or the fallback solver,
        so callers have one uniform return shape.
        """
        memory = self.memory_for(game_id)
        step_index = memory.step_index

        with self.resources.timed_step(game_id, step_index) as timing:
            source, decision = self._decide(game_id, action_space, complex_actions, memory)
            timing.used_fallback = source == "fallback"

        logger.debug("game=%s step=%d source=%s action=%s", game_id, step_index, source, decision.action)
        return decision

    def _decide(
        self,
        game_id: str,
        action_space: list[str],
        complex_actions: frozenset[str],
        memory: GameMemory,
    ) -> tuple[str, FallbackDecision]:
        if self.resources.should_use_fallback_only() or self.resources.game_time_exhausted(game_id):
            return "fallback", self.fallback.decide(action_space, memory, complex_actions)

        if memory.is_looping():
            return "fallback", self.fallback.decide(action_space, memory, complex_actions)

        prompt = memory.summarize_for_prompt()
        policy_output: PolicyOutput = self.policy.act(prompt, action_space)
        if policy_output.confidence < self.config.confidence_threshold:
            return "fallback", self.fallback.decide(action_space, memory, complex_actions)

        return "policy", FallbackDecision(
            policy_output.action, policy_output.action_args, reasoning="primary-policy"
        )

    def record(
        self,
        game_id: str,
        frame_before: MultiGrid,
        action: str,
        action_args: dict | None,
        frame_after: MultiGrid,
        game_state_after: str,
        levels_completed_after: int,
    ) -> None:
        self.memory_for(game_id).record(
            frame_before, action, action_args, frame_after, game_state_after, levels_completed_after
        )

    def shutdown(self) -> None:
        """Call once, after every game thread has finished (see resource_manager.py)."""
        self.policy.shutdown()
        self.resources.shutdown()


# ----------------------------------------------------------------------
# Process-wide singleton accessor. `agent/my_agent.py` calls this from
# `MyAgent.__init__`, which -- per the framework's Swarm -- runs once per
# game/thread. Double-checked locking so the (possibly slow, GPU-loading)
# HybridDecisionCore constructor only ever runs once.
# ----------------------------------------------------------------------
_core: HybridDecisionCore | None = None
_core_lock = threading.Lock()


def get_shared_core(config: AgentConfig | None = None) -> HybridDecisionCore:
    global _core
    if _core is not None:
        return _core
    with _core_lock:
        if _core is None:
            _core = HybridDecisionCore(config)
    return _core
