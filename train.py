"""Train a reward-driven TABasco DQN and save complete run metadata."""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch

from agent import DQNAgent
from environment import TabManagementEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=250)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=3090)
    parser.add_argument("--memory-budget", type=float, default=8.)
    parser.add_argument("--memory-coefficient", type=float, default=1.)
    parser.add_argument("--constraint-aware", action="store_true",
                        help="enforce active memory budget across actions and environment events")
    parser.add_argument("--n-tabs", type=int, default=12)
    parser.add_argument("--checkpoint", default="models/tabasco_dqn.pt")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=.97)
    parser.add_argument("--replay-capacity", type=int, default=50000)
    parser.add_argument("--target-update", type=int, default=500)
    parser.add_argument("--epsilon-start", type=float, default=1.)
    parser.add_argument("--epsilon-end", type=float, default=.05)
    parser.add_argument("--epsilon-decay", type=int, default=20000)
    parser.add_argument("--gradient-clip", type=float, default=10.)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    env = TabManagementEnv(args.n_tabs, args.memory_budget, args.max_steps, args.seed,
                           coefficients={"memory": args.memory_coefficient},
                           constraint_aware=args.constraint_aware)
    agent = DQNAgent(env.observation_space_shape[0], env.action_space_n,
                     seed=args.seed, learning_rate=args.learning_rate,
                     gamma=args.gamma, batch_size=args.batch_size,
                     replay_capacity=args.replay_capacity,
                     target_update=args.target_update,
                     epsilon_start=args.epsilon_start,
                     epsilon_end=args.epsilon_end,
                     epsilon_decay=args.epsilon_decay,
                     gradient_clip=args.gradient_clip)
    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "algorithm": "DQN", "device": "cpu",
              "observation_shape": env.observation_space_shape,
              "action_count": env.action_space_n,
              "network_hidden_units": [128, 128],
              "activation": "ReLU",
              "optimizer": "Adam",
              "loss": "Huber/smooth_l1",
              "gradient_clip": args.gradient_clip,
              "epsilon_start": args.epsilon_start,
              "epsilon_end": args.epsilon_end,
              "action_mask": ("occupied tab; SUSPEND only active; WAIT always; suspended KEEP only if it fits budget"
                              if args.constraint_aware else
                              "occupied tab; SUSPEND only active; WAIT always"),
              "environment_coefficients": env.c,
              "constraint_aware": env.constraint_aware,
              "training_episode_seeds": [args.seed + 1, args.seed + args.episodes]}
    (out / "training_config.json").write_text(json.dumps(config, indent=2))
    logs = []
    for episode in range(args.episodes):
        obs = env.reset(args.seed + episode + 1)
        episode_return = 0.
        losses = []
        memory = []
        violation_steps = 0
        restoration = 0.
        lost_work = 0.
        invalid_actions = 0
        for _ in range(args.max_steps):
            mask = env.action_mask(obs)
            action = agent.act(obs, explore=True, action_mask=mask)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.remember(obs, action, reward, next_obs, terminated or truncated,
                           next_action_mask=env.action_mask(next_obs))
            loss = agent.update()
            if loss is not None:
                losses.append(loss)
            obs = next_obs
            episode_return += reward
            memory.append(info["active_memory"])
            violation_steps += int(info["memory_violation"])
            restoration += info["restoration_cost"]
            lost_work += info["lost_work_cost"]
            invalid_actions += int(not info["action_valid"])
            if terminated or truncated:
                break
        logs.append({"episode": episode + 1, "episode_seed": args.seed + episode + 1,
                     "return": episode_return,
                     "mean_loss": float(np.mean(losses)) if losses else 0.,
                     "mean_active_memory": float(np.mean(memory)),
                     "peak_active_memory": max(memory, default=0.),
                     "memory_violation_steps": violation_steps,
                     "restoration_cost": restoration,
                     "lost_work_cost": lost_work,
                     "invalid_action_rate": invalid_actions / max(1, len(memory)),
                     "epsilon": agent.epsilon})
        interval = max(1, args.episodes // 10)
        if (episode + 1) % interval == 0:
            print(f"episode {episode + 1}/{args.episodes} return={episode_return:.2f} "
                  f"loss={logs[-1]['mean_loss']:.4f} epsilon={agent.epsilon:.3f}")

    with (out / "training.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=logs[0].keys())
        writer.writeheader()
        writer.writerows(logs)
    agent.save(args.checkpoint, config)
    print(f"Saved checkpoint: {args.checkpoint}")
    print(f"Mean return last 20: {np.mean([row['return'] for row in logs[-20:]]):.3f}")


if __name__ == "__main__":
    main()
