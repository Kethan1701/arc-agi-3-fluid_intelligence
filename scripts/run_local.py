#!/usr/bin/env python
"""
run_local.py
============
Local smoke-test / dev-loop runner: plays the agent against
`LocalMockEnv` for N episodes and prints a summary. This is the fastest
way to exercise state tracking, fallback routing, and resource
management end-to-end without any GPU or the real Kaggle harness.

Usage:
    python scripts/run_local.py --episodes 10
    python scripts/run_local.py --episodes 5 --use-vllm --model-dir ./models/my-model
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running as `python scripts/run_local.py` from repo root without
# installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.my_agent import ArcAgent, AgentConfig  # noqa: E402
from env.local_mock_env import LocalMockEnv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-vllm", action="store_true", help="use real VLLMPolicy instead of StubPolicy")
    parser.add_argument("--model-dir", default=None, help="local model directory (required if --use-vllm)")
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.use_vllm and not args.model_dir:
        parser.error("--use-vllm requires --model-dir")

    config = AgentConfig(
        model_dir=args.model_dir,
        use_vllm=args.use_vllm,
        confidence_threshold=args.confidence_threshold,
    )
    agent = ArcAgent(config)

    solved_count = 0
    for i in range(args.episodes):
        env = LocalMockEnv(seed=args.seed + i)
        solved = agent.play_game(env, game_id=f"local-game-{i}")
        solved_count += int(solved)

    print("\n=== Run summary ===")
    print(f"Episodes:        {args.episodes}")
    print(f"Solved:          {solved_count} ({solved_count / args.episodes:.0%})")
    print(f"Session solve rate (agent-tracked): {agent.session_memory.solve_rate:.2%}")
    print(f"Resource summary: {agent.resources.summary()}")

    agent.shutdown()


if __name__ == "__main__":
    main()
