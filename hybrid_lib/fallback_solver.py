"""
fallback_solver.py
===================
Deterministic, model-free solver used whenever:
    (a) the primary LLM policy reports confidence below threshold, or
    (b) the resource manager has forced fallback-only mode (low on time
        budget, or a VRAM/inference-time problem was detected), or
    (c) GameMemory.is_looping() detects the primary policy is stuck.

Design goals:
    * Zero GPU dependency -- must work even if vLLM/torch failed to load.
    * Fast -- this can run for many steps if the LLM is unavailable for
      the rest of the 9-hour budget.
    * Two layers, cheapest-first:
        1. Heuristic exploration: prefer actions that historically changed
           the frame, avoid the last action if it was a no-op, try
           never-before-seen actions to gather information.
        2. Tiny program-synthesis-style search: for grids small enough,
           brute-force short action sequences (depth-limited DFS/BFS) and
           replay the one that most reduces "distance" to a heuristic
           goal signal (e.g. levels_completed increasing, or grid stops
           changing in a way that matches a previously solved level's
           trajectory).

    This is intentionally simple -- a real ARC-AGI-3 fallback benefits far
    more from good exploration heuristics + safe defaults than from a
    hand-crafted grid DSL, since the game rules are unknown a priori and
    differ per game. Extend `_search_action_sequence` with real ARC-DSL
    primitives (rotate/flood-fill/symmetry-detection, etc.) if you want to
    invest further here; the hooks are structured so that's a drop-in.

This module has no dependency on `arcengine`/`GameAction` -- it operates
purely on action-name strings and a `GameMemory`. `agent/my_agent.py` is
responsible for translating `arcengine.GameAction`/`FrameData` to/from
this module's plain types.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from hybrid_lib.state import GameMemory

# arcengine.ComplexAction's x/y both range over [0, 63] (a fixed 64x64
# coordinate space) regardless of the actual game grid's size -- verified
# against arcengine.enums.ComplexAction. Actions that need this kind of
# (x, y) payload (e.g. ACTION6) are passed in via `complex_actions` below
# rather than hardcoded here, since that set is defined by arcengine, not
# by this module.
COMPLEX_ACTION_COORD_MAX = 63


@dataclass
class FallbackDecision:
    action: str
    action_args: dict | None
    reasoning: str


class FallbackSolver:
    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def decide(
        self,
        action_space: list[str],
        memory: GameMemory,
        complex_actions: frozenset[str] = frozenset(),
    ) -> FallbackDecision:
        action_space = action_space or ["RESET"]

        # 1) Prioritize actions we have never tried -- pure information
        #    gain, cheap, and often reveals game mechanics fast.
        untried = memory.untried_actions(action_space)
        if untried:
            action = self._rng.choice(untried)
            return self._finalize(action, complex_actions, reasoning="explore-untried-action")

        # 2) If we're in a detected loop, force a jump to whatever action
        #    has the highest historical "changed the frame" rate, biased
        #    away from the immediately preceding action to break cycles.
        if memory.is_looping():
            last_action = memory.recent_actions(1)
            ranked = sorted(
                action_space,
                key=lambda a: memory.action_effectiveness(a),
                reverse=True,
            )
            for candidate in ranked:
                if not last_action or candidate != last_action[0]:
                    return self._finalize(candidate, complex_actions, reasoning="break-detected-loop")

        # 3) Otherwise, softmax-ish weighted choice favoring actions that
        #    historically changed the frame (i.e. "did something").
        weights = [max(memory.action_effectiveness(a), 0.05) for a in action_space]
        total = sum(weights)
        probs = [w / total for w in weights]
        action = self._rng.choices(action_space, weights=probs, k=1)[0]
        return self._finalize(action, complex_actions, reasoning="weighted-exploration")

    def _finalize(
        self, action: str, complex_actions: frozenset[str], reasoning: str
    ) -> FallbackDecision:
        action_args = None
        if action in complex_actions:
            action_args = {
                "x": self._rng.randint(0, COMPLEX_ACTION_COORD_MAX),
                "y": self._rng.randint(0, COMPLEX_ACTION_COORD_MAX),
            }
        return FallbackDecision(action, action_args, reasoning=reasoning)

    # ------------------------------------------------------------------
    # Optional deeper search hook (disabled by default -- see class
    # docstring). Left as a documented extension point.
    # ------------------------------------------------------------------
    def _search_action_sequence(
        self,
        action_space: list[str],
        memory: GameMemory,
        max_depth: int = 4,
    ) -> list[str] | None:
        """
        Placeholder for a depth-limited search over short action sequences.
        Not wired into `decide()` by default because, without a scoring
        oracle for "closer to solved" beyond the game's own level-complete
        signal, blind lookahead search burns wall-clock budget with no
        guarantee of being better than the reactive heuristics above.
        Implement this once you've characterized a specific family of
        ARC-AGI-3 games (via `make play-local` against real games) where
        multi-step lookahead demonstrably helps.
        """
        raise NotImplementedError(
            "Wire up a scoring function specific to the games you're "
            "targeting before enabling sequence search."
        )
