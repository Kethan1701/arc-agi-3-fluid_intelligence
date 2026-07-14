"""
resource_manager.py
====================
Wall-clock safeguards for the 9-hour Kaggle runtime limit.

IMPORTANT -- concurrency model (verified against `agents/swarm.py` in the
official ARC-AGI-3-Agents framework): the framework's `Swarm.main()`
instantiates one `Agent` subclass PER GAME and runs ALL of them
*concurrently*, each in its own `Thread`, sharing one process. With up to
~110 games in a run, that means many `MyAgent.choose_action()` calls can
be in flight at once, from different threads, against whatever shared
model/resources this class holds.

Two consequences that shaped this design (and differ from a naive
"one game at a time" resource manager):
    1. Per-game timing state must be keyed by `game_id` in a dict, not
       stored as a single shared attribute -- multiple games are "in
       progress" simultaneously.
    2. There is NO safe point to call `torch.cuda.empty_cache()` /
       reload the model "between games", because there usually isn't a
       moment when *no* game is using the shared model -- other threads
       may have in-flight vLLM requests at any time. VRAM cleanup here is
       therefore a single explicit `shutdown()` call for process-exit
       only (see `hybrid_lib.decision_core.HybridDecisionCore`), not a
       per-game hook. Games sharing a single vLLM engine instance is
       actually the *correct* design for throughput (vLLM's continuous
       batching is built for concurrent requests) -- just don't tear
       anything down mid-run.

All public mutation methods are guarded by a single `threading.Lock`
for correctness under this concurrent-games model. This trades a little
throughput (lock contention is brief -- just bookkeeping, not the actual
model call) for avoiding races on shared counters/dicts.
"""

from __future__ import annotations

import contextlib
import gc
import threading
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
    Instantiate once per submission process (i.e. once for the whole 9h
    run) and share it across every concurrently-running game's agent.
    Thread-safe: see module docstring.
    """

    def __init__(self, budget: ResourceBudget | None = None) -> None:
        self.budget = budget or ResourceBudget()
        self._start_time = time.monotonic()
        self._lock = threading.Lock()
        self._game_start_times: dict[str, float] = {}
        self.step_log: list[StepTiming] = []
        self._forced_fallback_only = False
        self._force_reason: str | None = None

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
        with self._lock:
            self._forced_fallback_only = True
            self._force_reason = reason

    def game_time_exhausted(self, game_id: str) -> bool:
        with self._lock:
            start = self._game_start_times.get(game_id)
        if start is None:
            return False
        return (time.monotonic() - start) >= self.budget.max_seconds_per_game

    # ------------------------------------------------------------------
    # Per-game / per-step lifecycle hooks
    # ------------------------------------------------------------------
    def start_game(self, game_id: str) -> None:
        with self._lock:
            self._game_start_times[game_id] = time.monotonic()

    def end_game(self, game_id: str) -> None:
        """
        Call when one game finishes. Only clears that game's own timing
        entry -- does NOT touch GPU memory, since other games' threads
        may still be mid-inference against the same shared model. See
        module docstring / `HybridDecisionCore.shutdown()` for the actual
        (process-exit-only) VRAM cleanup path.
        """
        with self._lock:
            self._game_start_times.pop(game_id, None)

    @contextlib.contextmanager
    def timed_step(self, game_id: str, step_index: int):
        """
        Usage:
            with resource_manager.timed_step(game_id, step_idx) as timing:
                ... call the policy / fallback solver ...
                timing.used_fallback = <bool>

        Records duration regardless of success/failure. If the step ran
        over `max_seconds_per_step`, latches `force_fallback_only()` for
        all games for the rest of the run -- this cannot preemptively
        kill an in-flight vLLM call (that requires vLLM-level request
        cancellation/timeouts, configure those too for defense in depth),
        but it stops the *next* step across every game's thread from
        making the situation worse.
        """
        timing = StepTiming(game_id=game_id, step_index=step_index, duration_seconds=0.0, used_fallback=False)
        t0 = time.monotonic()
        try:
            yield timing
        finally:
            timing.duration_seconds = time.monotonic() - t0
            with self._lock:
                self.step_log.append(timing)
            if timing.duration_seconds > self.budget.max_seconds_per_step:
                self.force_fallback_only(
                    reason=f"step {step_index} of {game_id} took {timing.duration_seconds:.1f}s "
                    f"(> {self.budget.max_seconds_per_step}s budget)"
                )

    def shutdown(self) -> None:
        """
        Process-exit-only VRAM cleanup. Only safe to call once every
        game thread has finished (i.e. after `Swarm.main()` returns) --
        see module docstring for why this can't happen per-game.
        """
        gc.collect()
        _torch_cuda_cleanup()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def summary(self) -> dict:
        with self._lock:
            steps_recorded = len(self.step_log)
            active_games = len(self._game_start_times)
        return {
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "remaining_seconds": round(self.remaining_seconds, 1),
            "fallback_only": self.should_use_fallback_only(),
            "steps_recorded": steps_recorded,
            "active_games": active_games,
            "cuda_memory_mb": _torch_cuda_memory_mb(),
        }
