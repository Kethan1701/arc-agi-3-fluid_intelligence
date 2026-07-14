"""
Tests for `hybrid_lib`, run entirely offline against StubPolicy (no
GPU / vLLM required) and against real `arcengine.FrameData`/`GameAction`/
`GameState` objects (installed via the `arc-agi` pip package -- pure
Python + pydantic + numpy, no GPU dependency, works on Windows).

We deliberately do NOT instantiate `agent.my_agent.MyAgent` here: it
subclasses `agents.agent.Agent` from the separate `ARC-AGI-3-Agents`
framework, which is only importable once you've vendored that repo (see
README.md "Two-repo workflow"). Instead, these tests exercise
`HybridDecisionCore` directly with the same real `arcengine` types
`my_agent.py` translates to/from, which covers the actual decision logic
without needing the vendored framework installed in this repo.
"""

from __future__ import annotations

import threading

from arcengine import FrameData, GameAction, GameState

from hybrid_lib.decision_core import AgentConfig, HybridDecisionCore


def make_core(**overrides) -> HybridDecisionCore:
    return HybridDecisionCore(AgentConfig(use_vllm=False, **overrides))


def _frame(state: GameState, levels_completed: int = 0, marker_x: int = 0) -> FrameData:
    grid = [[0] * 4 for _ in range(4)]
    grid[0][marker_x % 4] = 5
    return FrameData(
        game_id="toy",
        frame=[grid],
        state=state,
        levels_completed=levels_completed,
        available_actions=[GameAction.ACTION1.value, GameAction.RESET.value],
    )


def play_toy_game(core: HybridDecisionCore, game_id: str, max_steps: int = 20) -> bool:
    """
    Drive a tiny synthetic game through `HybridDecisionCore`, mirroring
    what `agent/my_agent.py` does each `choose_action` call: consult the
    core for a decision, then record the resulting transition using real
    `arcengine` frame objects. ACTION1 moves a marker one cell right each
    step; the game is "solved" once the marker wraps past position 3.
    """
    core.start_game(game_id)
    action_space = [GameAction.ACTION1.name, GameAction.RESET.name]
    frame = _frame(GameState.NOT_PLAYED)
    solved = False
    for step in range(max_steps):
        decision = core.choose(game_id, action_space, complex_actions=frozenset())
        marker_x = step + 1
        new_state = GameState.WIN if marker_x >= 4 else GameState.NOT_FINISHED
        next_frame = _frame(new_state, levels_completed=int(new_state is GameState.WIN), marker_x=marker_x)
        core.record(
            game_id,
            frame.frame,
            decision.action,
            decision.action_args,
            next_frame.frame,
            game_state_after=next_frame.state.value,
            levels_completed_after=next_frame.levels_completed,
        )
        frame = next_frame
        if new_state is GameState.WIN:
            solved = True
            break
    core.end_game(game_id, solved=solved)
    return solved


def test_single_game_runs_without_exceptions():
    core = make_core()
    play_toy_game(core, "toy-1")
    core.shutdown()


def test_session_memory_accumulates_across_games():
    core = make_core()
    for i in range(3):
        play_toy_game(core, f"toy-{i}")
    assert core.session_memory.games_played == 3
    assert len(core.session_memory.per_game_step_counts) == 3
    core.shutdown()


def test_stub_policy_always_routes_to_fallback():
    """
    StubPolicy always reports confidence=0.0, which is below any sane
    threshold, so every decision should have gone through FallbackSolver
    rather than the (non-existent) primary model.
    """
    core = make_core(confidence_threshold=0.55)
    play_toy_game(core, "toy-fallback")

    assert len(core.resources.step_log) > 0
    assert all(step.used_fallback for step in core.resources.step_log)
    core.shutdown()


def test_game_memory_records_transitions():
    core = make_core()
    core.start_game("toy-memory")
    frame = _frame(GameState.NOT_PLAYED)
    next_frame = _frame(GameState.NOT_FINISHED, marker_x=1)

    core.record(
        "toy-memory",
        frame.frame,
        GameAction.ACTION1.name,
        None,
        next_frame.frame,
        game_state_after=next_frame.state.value,
        levels_completed_after=0,
    )

    memory = core.memory_for("toy-memory")
    assert memory.step_index == 1
    assert len(memory.history) == 1
    assert GameAction.ACTION1.name in memory.action_effect_log

    core.end_game("toy-memory", solved=False)
    core.shutdown()


def test_complex_action_gets_coordinate_args():
    """
    FallbackSolver must fill in (x, y) for actions the caller marks as
    complex (ACTION6 in the real game), since arcengine.ComplexAction
    requires that payload -- verified against arcengine.enums.
    """
    core = make_core()
    core.start_game("toy-complex")
    action_space = [GameAction.ACTION6.name]
    complex_actions = frozenset({GameAction.ACTION6.name})

    decision = core.choose("toy-complex", action_space, complex_actions)

    assert decision.action == GameAction.ACTION6.name
    assert decision.action_args is not None
    assert 0 <= decision.action_args["x"] <= 63
    assert 0 <= decision.action_args["y"] <= 63
    core.end_game("toy-complex", solved=False)
    core.shutdown()


def test_concurrent_games_do_not_corrupt_shared_state():
    """
    The real harness runs every game's agent concurrently in its own
    thread against one shared HybridDecisionCore (verified against
    agents/swarm.py in ARC-AGI-3-Agents). This smoke-tests that N games
    played on separate threads against one shared core don't crash or
    cross-contaminate each other's step counts.
    """
    core = make_core()
    n_games = 8
    results: dict[str, bool] = {}

    def run(i: int) -> None:
        game_id = f"toy-concurrent-{i}"
        results[game_id] = play_toy_game(core, game_id, max_steps=10)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(n_games)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert core.session_memory.games_played == n_games
    assert len(results) == n_games
    core.shutdown()
