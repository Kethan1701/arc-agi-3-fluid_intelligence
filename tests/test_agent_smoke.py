"""
Smoke tests for the full agent loop, run entirely offline with
StubPolicy (no GPU / vLLM required) against LocalMockEnv.

These intentionally don't assert on *solve quality* -- LocalMockEnv's
push-to-origin puzzle happens to be simple enough that the fallback
solver's weighted exploration solves it reliably, but that's a property
of this particular toy env, not a guarantee for arbitrary ARC-AGI-3
games. The tests instead assert the pipeline runs end-to-end without
exceptions and that the hybrid routing, memory, and resource bookkeeping
behave as designed.
"""

from __future__ import annotations

from agent.my_agent import AgentConfig, ArcAgent
from env.local_mock_env import LocalMockEnv


def make_agent(**overrides) -> ArcAgent:
    config = AgentConfig(use_vllm=False, max_steps_per_game=50, **overrides)
    return ArcAgent(config)


def test_single_game_runs_without_exceptions():
    agent = make_agent()
    env = LocalMockEnv(seed=1)
    agent.play_game(env, game_id="smoke-1")
    agent.shutdown()


def test_session_memory_accumulates_across_games():
    agent = make_agent()
    for i in range(3):
        env = LocalMockEnv(seed=i)
        agent.play_game(env, game_id=f"smoke-{i}")
    assert agent.session_memory.games_played == 3
    assert len(agent.session_memory.per_game_step_counts) == 3
    agent.shutdown()


def test_stub_policy_always_routes_to_fallback():
    """
    StubPolicy always reports confidence=0.0, which is below any sane
    threshold, so every single decision this episode should have gone
    through FallbackSolver rather than the (non-existent) primary model.
    We verify this indirectly: resource_manager's step_log should show
    used_fallback=True for every recorded step.
    """
    agent = make_agent(confidence_threshold=0.55)
    env = LocalMockEnv(seed=42)
    agent.play_game(env, game_id="smoke-fallback")

    assert len(agent.resources.step_log) > 0
    assert all(step.used_fallback for step in agent.resources.step_log)
    agent.shutdown()


def test_game_memory_records_transitions():
    agent = make_agent()
    env = LocalMockEnv(seed=7, size=6)
    observation = env.reset("smoke-memory")
    agent.start_game("smoke-memory")

    frame_before = observation.frame
    action, action_args = agent.act(observation)
    observation = env.step(action, action_args)
    agent.record_transition(frame_before, action, action_args, observation)

    memory = agent._memory  # noqa: SLF001 -- intentional white-box check
    assert memory is not None
    assert memory.step_index == 1
    assert len(memory.history) == 1
    assert action in memory.action_effect_log

    agent.end_game(solved=observation.level_complete)
    agent.shutdown()
