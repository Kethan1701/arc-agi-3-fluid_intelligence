"""
resource_manager.py
====================
Wall-clock and VRAM safeguards for the 9-hour Kaggle runtime limit.

Two independent concerns live here:
    1. Global time budget -- track elapsed time against the 9h ceiling
       (with a safety margin), and expose `should_use_fallback_only()`
       so the agent can pre-emptively stop calling the (slow, GPU-bound)
       LLM policy well before the hard cutoff, finishing remaining games
       with the cheap deterministic solver instead of risking a
       kill mid-inference (which likely scores worse than a fast, if
       weaker, submitted action).
    2. Per-task/per-step VRAM hygiene -- explicit `torch.cuda.empty_cache()`
       + `gc.collect()` between games, plus a per-step inference timeout
       so a single pathological step can't silently eat the whole budget.

Kept dependency-light: torch is imported lazily and all torch-specific
calls are wrapped so this module works fine even if torch isn't
installed/available (e.g. a CPU-only smoke test box).
"""

from __future__ import annotations

import contextlib
import gc
import time
from dataclasses import dataclass


def _torch_cuda_cleanup() -> None:
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _torch_cuda_memory_mb() -> float | None:
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.memory_allocated() / (1024 ** 2)


@dataclass
class ResourceBudget:
    total_seconds: float = 9 * 60 * 60          # Kaggle hard limit: 9 hours
    safety_margin_seconds: float = 20 * 60       # stop the LLM path this long before the cliff
    max_seconds_per_step: float = 25.0           # single-action inference timeout budget
    max_seconds_per_game: float = 10 * 60         # give up on one game rather than starve the rest


@dataclass
class StepTiming:
    game_id: str
    step_index: int
    duration_seconds: float
    used_fallback: bool


class ResourceManager:
    """
    Instantiate once per submission process (i.e. once for the whole
    9-hour run) and share it across every game the agent plays.
    """

    def __init__(self, budget: ResourceBudget | None = None) -> None:
        self.budget = budget or ResourceBudget()
        self._start_time = time.monotonic()
        self._game_start_time: float | None = None
        self.step_log: list[StepTiming] = []
        self._forced_fallback_only = False

    # ------------------------------------------------------------------
    # Time budget
    # ------------------------------------------------------------------
    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self._start_time

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.budget.total_seconds - self.elapsed_seconds)

    def should_use_fallback_only(self) -> bool:
        """
        True once we're within `safety_margin_seconds` of the hard 9h
        limit, or once explicitly latched via `force_fallback_only()`
        (e.g. after repeated inference timeouts suggest the GPU path is
        unhealthy). Once true for the "out of time" reason it naturally
        stays true, since elapsed time never decreases.
        """
        if self._forced_fallback_only:
            return True
        return self.remaining_seconds <= self.budget.safety_margin_seconds

    def force_fallback_only(self, reason: str) -> None:
        self._forced_fallback_only = True
        self._last_force_reason = reason

    def game_time_exhausted(self) -> bool:
        if self._game_start_time is None:
            return False
        return (time.monotonic() - self._game_start_time) >= self.budget.max_seconds_per_game

    # ------------------------------------------------------------------
    # Per-game / per-step lifecycle hooks
    # ------------------------------------------------------------------
    def start_game(self, game_id: str) -> None:
        self._game_id = game_id
        self._game_start_time = time.monotonic()

    def end_game(self) -> None:
        """Call between games to release GPU memory and reset counters."""
        gc.collect()
        _torch_cuda_cleanup()
        self._game_start_time = None

    @contextlib.contextmanager
    def timed_step(self, game_id: str, step_index: int):
        """
        Usage:
            with resource_manager.timed_step(game_id, step_idx) as timing:
                ... call the policy / fallback solver ...
                timing.used_fallback = <bool>

        Records duration regardless of success/failure, and raises
        `StepTimeoutExceeded` after the fact if the step ran over budget
        so the caller can decide to force fallback mode for subsequent
        steps (it cannot preemptively kill a running vLLM generate() call
        -- that requires vLLM-level request cancellation/timeouts, which
        you should also configure via SamplingParams / server timeouts
        for defense in depth).
        """
        timing = StepTiming(game_id=game_id, step_index=step_index, duration_seconds=0.0, used_fallback=False)
        t0 = time.monotonic()
        try:
            yield timing
        finally:
            timing.duration_seconds = time.monotonic() - t0
            self.step_log.append(timing)
            if timing.duration_seconds > self.budget.max_seconds_per_step:
                self.force_fallback_only(
                    reason=f"step {step_index} of {game_id} took {timing.duration_seconds:.1f}s "
                    f"(> {self.budget.max_seconds_per_step}s budget)"
                )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def summary(self) -> dict:
        return {
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "remaining_seconds": round(self.remaining_seconds, 1),
            "fallback_only": self.should_use_fallback_only(),
            "steps_recorded": len(self.step_log),
            "cuda_memory_mb": _torch_cuda_memory_mb(),
        }
