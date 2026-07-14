"""
env_adapter.py
==============
Thin translation layer between our agent and the ARC-AGI-3 game harness.

IMPORTANT / HONESTY NOTE FOR THE DEVELOPER:
--------------------------------------------
The exact function signatures the Kaggle grading harness calls on your
submitted agent (and the exact shape of observations/actions it hands you)
are defined by the official ARC-AGI-3 starter kit, which you should pull
down from the competition's "Code" / "Data" tab once you have accepted the
rules (internet access is required for that one-time download; the
inference environment itself is offline).

Rather than guess at private/unstable field names, this module defines a
small, stable *internal* Protocol (`ArcEnv`) that the rest of our pipeline
(agent, state tracker, fallback solver) is built against. You only need to
edit `KaggleHarnessEnv` below -- a thin shim -- once you have the real
starter kit in hand. Everything else in this repo is harness-agnostic and
will not need to change.

Known, stable facts about ARC-AGI-3 (from the public competition
description) that this scaffold assumes:
    * Each "game" is a sequence of one or more levels the agent plays
      interactively, not a single static input/output grid.
    * Observations are 2D grids of small integers (colors), similar to
      ARC-AGI-1/2, but arrive as a *stream* of frames after each action.
    * The agent chooses discrete actions step-by-step and receives
      feedback (new frame, score/level-complete signal, done flag).
    * A game ends on success, failure, or a step/time budget being hit.

If the real harness differs (e.g. richer action schema, multiple frames
per step, extra metadata), update `RawObservation`/`ArcEnv`/
`KaggleHarnessEnv` -- the dataclasses are intentionally permissive
(`extra: dict`) so you can carry undocumented fields through without
breaking type contracts elsewhere.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Protocol


Grid = list[list[int]]  # a 2D array of small integer color codes


@dataclass
class RawObservation:
    """Normalized observation returned after `reset()` / `step()`."""

    frame: Grid                       # current game grid
    score: float = 0.0                # cumulative score / reward signal
    done: bool = False                 # episode (game) finished
    level_complete: bool = False       # sub-level finished, game continues
    action_space: list[str] = field(default_factory=list)  # legal action ids this step
    extra: dict[str, Any] = field(default_factory=dict)     # passthrough for harness-specific fields


class ArcEnv(Protocol):
    """Minimal Gym-like interface our agent is written against."""

    def reset(self, game_id: str) -> RawObservation: ...

    def step(self, action: str, action_args: dict[str, Any] | None = None) -> RawObservation: ...

    @property
    def game_id(self) -> str: ...


class KaggleHarnessEnv(abc.ABC):
    """
    Shim to be filled in once you have the official starter kit.

    Typical pattern for these Kaggle "agent plays a live env" competitions
    is that your submitted module exposes a top-level callable (commonly
    `agent(observation, configuration)` or a class with `.act()`) that the
    harness process imports and drives directly -- i.e. *you* usually will
    NOT be calling reset()/step() yourself in the submission; the harness
    calls INTO your agent. Keep this class around for local testing against
    a recorded/replayed trace of real games, and wire the real entrypoint
    (see agent/my_agent.py: `agent_entrypoint`) to your `ArcAgent` once the
    starter kit's exact calling convention is confirmed.
    """

    def __init__(self, connection: Any) -> None:
        self._conn = connection
        self._game_id: str | None = None

    @property
    def game_id(self) -> str:
        assert self._game_id is not None, "call reset() first"
        return self._game_id

    @abc.abstractmethod
    def reset(self, game_id: str) -> RawObservation:
        """Translate the harness's reset call into a RawObservation."""

    @abc.abstractmethod
    def step(self, action: str, action_args: dict[str, Any] | None = None) -> RawObservation:
        """Translate the harness's step call into a RawObservation."""
