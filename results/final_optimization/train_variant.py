"""Train a reward-weight variant with TABasco's existing DQN and environment.

This experiment-local script writes only beneath results/final_optimization.
The DQN implementation is imported unchanged from the repository root.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agent import DQNAgent  # noqa: E402
from environment import TabManagementEnv  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lost", type=float, default=1.2)
    parser.add_argument("--restore", type=float, default=.8)
    parser.add_argument("--disruption", type=float, default=.45)
    parser.add_argument("--episodes", type=int, default=250)
    parser.add_argument("--max-steps", type=int, default=120)
    args = parser.parse_args()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    coefficients = {"memory": 1., "restore": args.restore, "lost": args.lost,
                    "disruption": args.disruption, "action": .035,
                    "invalid": .12, "deferred": .25}
    env = TabManagementEnv(n_tabs=12, memory_budget=8., max_steps=args.max_steps,
                          seed=seed, coefficients=coefficients,
                          constraint_aware=True)
    agent = DQNAgent(env.observation_space_shape[0], env.action_space_n, seed=seed,
                     learning_rate=.001, gamma=.97, batch_size=64,
                     replay_capacity=50000, target_update=500,
                     epsilon_start=1., epsilon_end=.05, epsilon_decay=20000,
                     gradient_clip=10.)
    run_dir = ROOT / "results" / "final_optimization" / "runs" / args.name / f"seed{seed}"
    checkpoint = ROOT / "results" / "final_optimization" / "checkpoints" / f"{args.name}_seed{seed}.pt"
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {"name": args.name, "episodes": args.episodes,
              "max_steps": args.max_steps, "seed": seed,
              "memory_budget": 8., "n_tabs": 12,
              "constraint_aware": True,
              "coefficients": {**coefficients},
              "learning_rate": .001, "gamma": .97, "batch_size": 64,
              "replay_capacity": 50000, "target_update": 500,
              "epsilon_start": 1., "epsilon_end": .05,
              "epsilon_decay": 20000, "gradient_clip": 10.,
              "architecture": [101, 128, 128, 37], "device": "cpu",
              "checkpoint": str(checkpoint.relative_to(ROOT)),
              "training_episode_seeds": [seed + 1, seed + args.episodes],
              "action_mask": "environment mask used for exploration, greedy actions, and stored next-state target mask"}
    (run_dir / "training_config.json").write_text(json.dumps(config, indent=2))
    logs = []
    started = time.perf_counter()
    for episode in range(args.episodes):
        obs = env.reset(seed + episode + 1)
        total = 0.
        losses = []
        memory = []
        violations = 0
        restoration = lost = disruption = deferred = 0.
        invalid = 0
        for _ in range(args.max_steps):
            action = agent.act(obs, explore=True, action_mask=env.action_mask(obs))
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.remember(obs, action, reward, next_obs, terminated or truncated,
                           next_action_mask=env.action_mask(next_obs))
            loss = agent.update()
            if loss is not None:
                losses.append(loss)
            obs = next_obs
            total += reward
            memory.append(info["active_memory"])
            violations += int(info["memory_violation"])
            restoration += info["restoration_cost"]
            lost += info["lost_work_cost"]
            disruption += info["task_disruption_cost"]
            deferred += info["deferred_event_cost"]
            invalid += int(not info["action_valid"])
            if terminated or truncated:
                break
        logs.append({"episode": episode + 1, "episode_seed": seed + episode + 1,
                     "return": total, "mean_loss": float(np.mean(losses)) if losses else 0.,
                     "mean_active_memory": float(np.mean(memory)),
                     "peak_active_memory": max(memory, default=0.),
                     "violation_steps": violations,
                     "restoration_cost": restoration, "lost_work_cost": lost,
                     "disruption_cost": disruption, "deferred_cost": deferred,
                     "invalid_action_rate": invalid / max(1, len(memory)),
                     "epsilon": agent.epsilon})
        if (episode + 1) % max(1, args.episodes // 10) == 0:
            print(f"{args.name} seed={seed} episode={episode + 1}/{args.episodes} "
                  f"return={total:.2f} loss={logs[-1]['mean_loss']:.4f}", flush=True)
    elapsed = time.perf_counter() - started
    with (run_dir / "training.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(logs[0]))
        writer.writeheader()
        writer.writerows(logs)
    config["runtime_seconds"] = elapsed
    (run_dir / "training_config.json").write_text(json.dumps(config, indent=2))
    agent.save(checkpoint, config)
    print(f"saved={checkpoint} runtime_seconds={elapsed:.2f} "
          f"last20_return={np.mean([x['return'] for x in logs[-20:]]):.3f}", flush=True)


if __name__ == "__main__":
    main()
