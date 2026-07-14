"""
state.py
========
Interactive state/memory tracking for a single ARC-AGI-3 game.

ARC-AGI-3 tasks are *sequential decision processes*, not one-shot
input->output transforms. The agent needs a running memory of:
    * every (observation, action, resulting observation) transition,
      so the LLM's prompt can include recent history for planning;
    * a compact "world model" -- a dict of hypotheses/observations about
      how actions affect the grid, built incrementally, which the
      deterministic fallback solver also reads and writes;
    * loop/stagnation detection, since a common interactive-agent failure
      mode is repeating an action (or short cycle of actions) forever
      without making progress.

This module has zero framework dependencies (no torch/vllm) so it stays
cheap to import and easy to unit test.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from env.env_adapter import Grid, RawObservation


@dataclass
class Transition:
    """One (state, action, next_state) step, kept for history/replay."""

    step_index: int
    frame_before: Grid
    action: str
    action_args: dict[str, Any] | None
    frame_after: Grid
    score_after: float
    done: bool
    level_complete: bool


class GameMemory:
    """
    Per-game episodic memory + lightweight world model.

    One instance should be created per game (via `reset()`), and reused
    across every step of that game so the agent can reason about history.
    Keep it separate from `ResourceManager` (agent/resource_manager.py),
    which tracks *time/compute* budget rather than *task* state.
    """

    def __init__(self, game_id: str, max_history: int = 200, loop_window: int = 6) -> None:
        self.game_id = game_id
        self.max_history = max_history
        self.loop_window = loop_window

        self.history: deque[Transition] = deque(maxlen=max_history)
        self.step_index: int = 0

        # world_model: coarse, hand-inspectable facts the agent has learned
        # this episode, e.g. {"UP_changes_frame": True, "RIGHT_is_noop_at_edge": True}
        self.world_model: dict[str, Any] = {}

        # action -> list of "did this action change the frame?" booleans,
        # used by the fallback solver to bias exploration toward actions
        # that are known to actually do something.
        self.action_effect_log: dict[str, list[bool]] = {}

        self.last_observation: RawObservation | None = None

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def record(
        self,
        frame_before: Grid,
        action: str,
        action_args: dict[str, Any] | None,
        obs_after: RawObservation,
    ) -> None:
        transition = Transition(
            step_index=self.step_index,
            frame_before=frame_before,
            action=action,
            action_args=action_args,
            frame_after=obs_after.frame,
            score_after=obs_after.score,
            done=obs_after.done,
            level_complete=obs_after.level_complete,
        )
        self.history.append(transition)
        self.step_index += 1
        self.last_observation = obs_after

        changed = frame_before != obs_after.frame
        self.action_effect_log.setdefault(action, []).append(changed)

    # ------------------------------------------------------------------
    # Queries used by the policy / fallback solver
    # ------------------------------------------------------------------
    def action_effectiveness(self, action: str) -> float:
        """
        Fraction of times this action has changed the frame historically.
        Unseen actions return 1.0 (optimistic) so they get tried at least
        once -- standard optimistic-initialization exploration bias.
        """
        log = self.action_effect_log.get(action)
        if not log:
            return 1.0
        return sum(log) / len(log)

    def is_looping(self) -> bool:
        """
        Detect whether the last `loop_window` actions form a repeating
        cycle that hasn't changed the frame at all -- a strong signal the
        primary policy is stuck and control should shift to the
        fallback/exploration solver.
        """
        if len(self.history) < self.loop_window:
            return False
        recent = list(self.history)[-self.loop_window :]
        frames_unique = {tuple(map(tuple, t.frame_after)) for t in recent}
        return len(frames_unique) == 1

    def recent_actions(self, n: int = 10) -> list[str]:
        return [t.action for t in list(self.history)[-n:]]

    def untried_actions(self, action_space: list[str]) -> list[str]:
        return [a for a in action_space if a not in self.action_effect_log]

    def summarize_for_prompt(self, n: int = 8) -> str:
        """
        Compact text summary of recent history for the LLM prompt. Keeping
        this small matters: every extra token here is repeated on every
        step of a potentially long-running game, and vLLM throughput /
        our 9-hour wall-clock budget both benefit from short prompts.
        """
        lines = []
        for t in list(self.history)[-n:]:
            outcome = "changed" if t.frame_before != t.frame_after else "no-op"
            lines.append(f"step={t.step_index} action={t.action} -> {outcome}, score={t.score_after:.2f}")
        return "\n".join(lines) if lines else "(no history yet)"


class SessionMemory:
    """
    Cross-game memory for the lifetime of the whole submission process
    (i.e. across the entire 9-hour run, potentially many games/levels).

    Useful for transferring cheap signal between tasks without persisting
    full per-game history: e.g. "on grids of this size, action X is
    usually the reset/no-op action" or aggregate timing stats consumed by
    the resource manager.
    """

    def __init__(self) -> None:
        self.games_played: int = 0
        self.games_solved: int = 0
        self.per_game_step_counts: list[int] = []
        self.global_action_effect_log: dict[str, list[bool]] = {}

    def ingest(self, memory: GameMemory, solved: bool) -> None:
        self.games_played += 1
        self.games_solved += int(solved)
        self.per_game_step_counts.append(memory.step_index)
        for action, log in memory.action_effect_log.items():
            self.global_action_effect_log.setdefault(action, []).extend(log)

    @property
    def solve_rate(self) -> float:
        return self.games_solved / self.games_played if self.games_played else 0.0
