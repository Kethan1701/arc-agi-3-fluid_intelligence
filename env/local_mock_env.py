"""
local_mock_env.py
==================
A tiny, self-contained fake ARC-AGI-3 environment so you can develop and
smoke-test the agent loop (state tracking, fallback solver, resource
manager) entirely offline, without the real competition harness.

This is NOT meant to be a faithful simulation of actual ARC-AGI-3 games --
it just needs to satisfy the `ArcEnv` protocol so the rest of the pipeline
can be exercised end-to-end locally before you ever touch Kaggle.

Behavior: an agent occupies a cell on an NxN grid and must push a single
"goal" tile (indicated by color 9) onto the origin by moving. Each move
shifts every "movable" cell by one step in the chosen direction. This is
enough non-determinism/interactivity to test exploration and memory logic.
"""

from __future__ import annotations

import random

from env.env_adapter import ArcEnv, Grid, RawObservation

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "RESET", "NOOP"]


class LocalMockEnv(ArcEnv):
    def __init__(self, size: int = 8, seed: int | None = None) -> None:
        self.size = size
        self._rng = random.Random(seed)
        self._game_id = "local-mock"
        self._steps_taken = 0
        self._max_steps = 80
        self._goal_pos = (0, 0)
        self._agent_pos = (0, 0)

    @property
    def game_id(self) -> str:
        return self._game_id

    def reset(self, game_id: str) -> RawObservation:
        self._game_id = game_id
        self._steps_taken = 0
        self._agent_pos = (self._rng.randrange(self.size), self._rng.randrange(self.size))
        self._goal_pos = (self._rng.randrange(self.size), self._rng.randrange(self.size))
        return RawObservation(
            frame=self._render(),
            score=0.0,
            done=False,
            level_complete=False,
            action_space=ACTIONS,
        )

    def step(self, action: str, action_args: dict | None = None) -> RawObservation:
        self._steps_taken += 1
        ax, ay = self._agent_pos

        if action == "RESET":
            return self.reset(self._game_id)
        elif action == "UP":
            ay = max(0, ay - 1)
        elif action == "DOWN":
            ay = min(self.size - 1, ay + 1)
        elif action == "LEFT":
            ax = max(0, ax - 1)
        elif action == "RIGHT":
            ax = min(self.size - 1, ax + 1)
        # NOOP / unrecognized action: no movement

        self._agent_pos = (ax, ay)
        solved = self._agent_pos == self._goal_pos
        timed_out = self._steps_taken >= self._max_steps

        return RawObservation(
            frame=self._render(),
            score=1.0 if solved else 0.0,
            done=solved or timed_out,
            level_complete=solved,
            action_space=ACTIONS,
            extra={"steps_taken": self._steps_taken, "timed_out": timed_out},
        )

    def _render(self) -> Grid:
        grid = [[0 for _ in range(self.size)] for _ in range(self.size)]
        gx, gy = self._goal_pos
        ax, ay = self._agent_pos
        grid[gy][gx] = 9
        grid[ay][ax] = 5 if (ax, ay) != (gx, gy) else 3
        return grid
