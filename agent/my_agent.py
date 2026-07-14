"""
my_agent.py
===========
Main ARC-AGI-3 agent: wires together the offline model policy
(model_runtime.py), episodic state/memory (state.py), the deterministic
fallback solver (fallback_solver.py), and the resource manager
(resource_manager.py) into a single decision loop.

Routing logic per step (the "hybrid" in "hybrid fallback system"):
    1. If the resource manager says we're out of time budget -> fallback.
    2. If GameMemory detects we're stuck in a loop -> fallback (break it).
    3. Otherwise, ask the primary policy (VLLMPolicy / StubPolicy).
       If its self-reported confidence < CONFIG.confidence_threshold,
       or it returned an illegal/unparseable action -> fallback.
    4. Record the transition in memory either way, so both the LLM
       policy's future prompts and the fallback solver's heuristics
       benefit from what was learned this step.

This file intentionally has NO import-time dependency on vLLM/torch
actually being installed -- `build_policy()` decides which backend to
construct based on config, so you can run the exact same `ArcAgent` class
locally (StubPolicy, no GPU) and on Kaggle (VLLMPolicy, real GPU).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from agent.fallback_solver import FallbackSolver
from agent.model_runtime import Policy, PolicyOutput, StubPolicy
from agent.resource_manager import ResourceManager
from agent.state import GameMemory, SessionMemory
from env.env_adapter import ArcEnv, RawObservation

logger = logging.getLogger("arc_agent")


@dataclass
class AgentConfig:
    model_dir: str | None = None          # local path; None => use StubPolicy (no GPU)
    confidence_threshold: float = 0.55     # below this, route to fallback solver
    max_steps_per_game: int = 200
    use_vllm: bool = False                 # explicit opt-in, see build_policy()


def build_policy(config: AgentConfig) -> Policy:
    """
    Construct the primary policy backend. Kept as a free function (rather
    than inline in `ArcAgent.__init__`) so tests / local runs can swap in
    a fake Policy without touching ArcAgent itself.
    """
    if not config.use_vllm or not config.model_dir:
        logger.info("Using StubPolicy (no GPU model loaded) - set config.use_vllm=True and "
                     "config.model_dir=<local path> to enable real inference.")
        return StubPolicy()

    # Imported lazily via model_runtime; will raise a clear error locally
    # (e.g. on Windows without vLLM) rather than at module import time.
    from agent.model_runtime import VLLMPolicy  # noqa: PLC0415

    return VLLMPolicy(model_dir=config.model_dir)


class ArcAgent:
    """
    One instance is created for the whole submission process and reused
    across every game the harness presents (see `agent_entrypoint` at the
    bottom of this file for the Kaggle-facing hook). Keeping one instance
    means the model weights are loaded exactly once, and the
    `ResourceManager`'s 9-hour clock starts at construction, matching the
    real runtime constraint.
    """

    def __init__(self, config: AgentConfig | None = None) -> None:
        self.config = config or AgentConfig()
        self.policy: Policy = build_policy(self.config)
        self.fallback = FallbackSolver()
        self.resources = ResourceManager()
        self.session_memory = SessionMemory()
        self._memory: GameMemory | None = None

    # ------------------------------------------------------------------
    # Per-game lifecycle
    # ------------------------------------------------------------------
    def start_game(self, game_id: str) -> None:
        self._memory = GameMemory(game_id=game_id)
        self.resources.start_game(game_id)
        logger.info("Starting game %s | budget=%s", game_id, self.resources.summary())

    def end_game(self, solved: bool) -> None:
        assert self._memory is not None
        self.session_memory.ingest(self._memory, solved=solved)
        self.resources.end_game()  # VRAM cleanup + reset per-game timer
        logger.info(
            "Ended game %s solved=%s steps=%d | session_solve_rate=%.2f",
            self._memory.game_id,
            solved,
            self._memory.step_index,
            self.session_memory.solve_rate,
        )
        self._memory = None

    # ------------------------------------------------------------------
    # Core decision step
    # ------------------------------------------------------------------
    def act(self, observation: RawObservation) -> tuple[str, dict | None]:
        """
        Choose the next action given the current observation. Returns
        (action, action_args). This is the function you call once per
        environment step; the resulting (action, next_observation) pair
        should be fed back via `record_transition` below.
        """
        assert self._memory is not None, "call start_game() first"
        memory = self._memory
        step_index = memory.step_index

        with self.resources.timed_step(memory.game_id, step_index) as timing:
            decision_source, action, action_args = self._decide(observation, memory)
            timing.used_fallback = decision_source == "fallback"

        logger.debug("step=%d source=%s action=%s", step_index, decision_source, action)
        return action, action_args

    def _decide(
        self, observation: RawObservation, memory: GameMemory
    ) -> tuple[str, str, dict | None]:
        # 1) Hard resource ceiling reached -> always fallback, regardless
        #    of what the primary policy would have said.
        if self.resources.should_use_fallback_only() or self.resources.game_time_exhausted():
            decision = self.fallback.decide(observation, memory)
            return "fallback", decision.action, decision.action_args

        # 2) Detected a stuck loop -> break it deterministically rather
        #    than trusting the model to notice its own repetition.
        if memory.is_looping():
            decision = self.fallback.decide(observation, memory)
            return "fallback", decision.action, decision.action_args

        # 3) Ask the primary policy; fall back on low confidence.
        prompt = memory.summarize_for_prompt()
        policy_output: PolicyOutput = self.policy.act(prompt, observation.action_space)
        if policy_output.confidence < self.config.confidence_threshold:
            decision = self.fallback.decide(observation, memory)
            return "fallback", decision.action, decision.action_args

        return "policy", policy_output.action, policy_output.action_args

    def record_transition(
        self,
        frame_before,
        action: str,
        action_args: dict | None,
        obs_after: RawObservation,
    ) -> None:
        assert self._memory is not None
        self._memory.record(frame_before, action, action_args, obs_after)

    # ------------------------------------------------------------------
    # Convenience: drive a full local ArcEnv-compatible env end-to-end.
    # Handy for local_mock_env.py testing; NOT used by the Kaggle harness
    # path, which instead calls `act`/`record_transition` step-by-step
    # under harness control (see agent_entrypoint below).
    # ------------------------------------------------------------------
    def play_game(self, env: ArcEnv, game_id: str) -> bool:
        observation = env.reset(game_id)
        self.start_game(game_id)
        solved = False
        for _ in range(self.config.max_steps_per_game):
            frame_before = observation.frame
            action, action_args = self.act(observation)
            observation = env.step(action, action_args)
            self.record_transition(frame_before, action, action_args, observation)
            if observation.done:
                solved = observation.level_complete
                break
        self.end_game(solved)
        return solved

    def shutdown(self) -> None:
        self.policy.shutdown()


# ----------------------------------------------------------------------
# Kaggle-facing entrypoint.
#
# NOTE: the exact calling convention the grader uses (module-level
# function vs. class instantiation, argument names/types) is defined by
# the official ARC-AGI-3 starter kit -- confirm this against the real
# starter notebook before your first submission and adjust the signature
# below accordingly. A single module-level `AGENT` singleton is used here
# so model weights load once per process rather than once per game.
# ----------------------------------------------------------------------
AGENT: ArcAgent | None = None


def get_agent() -> ArcAgent:
    global AGENT
    if AGENT is None:
        AGENT = ArcAgent(
            AgentConfig(
                model_dir="/kaggle/input/your-model-dataset/model",  # EDIT ME
                use_vllm=True,
                confidence_threshold=0.55,
            )
        )
    return AGENT
