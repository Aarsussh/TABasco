"""Run one deterministic, readable TABasco episode with a trained constrained DQN."""
import argparse
from pathlib import Path

import numpy as np

from agent import DQNAgent
from environment import CLOSE, KEEP, SUSPEND, WAIT, TabManagementEnv


def describe_action(action, env):
    index, operation = env.decode_action(action)
    if operation == WAIT:
        return "WAIT (no eligible tab operation)"
    tab = env.tabs[index]
    if operation == KEEP:
        if tab["active"]:
            return f"KEEP tab {index} (already active)"
        return f"RESTORE tab {index}"
    if operation == SUSPEND:
        return f"SUSPEND tab {index}"
    if operation == CLOSE:
        return f"CLOSE tab {index}"
    return f"invalid action {action}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=3090,
                        help="environment episode seed (default: 3090)")
    parser.add_argument("--model-seed", type=int, choices=(3090, 4090), default=4090,
                        help="training seed for the selected constrained checkpoint")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="optional checkpoint override")
    parser.add_argument("--steps", type=int, default=40,
                        help="episode horizon; use 120 to match the training horizon")
    args = parser.parse_args()

    default_checkpoint = Path("models/sensitivity") / f"constrained_seed{args.model_seed}.pt"
    checkpoint = args.checkpoint or default_checkpoint
    if not checkpoint.is_file():
        command = (f"python train.py --episodes 250 --max-steps 120 --seed {args.model_seed} "
                   f"--constraint-aware --checkpoint {checkpoint} "
                   f"--results-dir results/demo_train_seed{args.model_seed}")
        raise SystemExit(f"Checkpoint not found: {checkpoint}\nTrain it with:\n{command}")

    np.set_printoptions(precision=2, suppress=True)
    env = TabManagementEnv(n_tabs=12, memory_budget=8., max_steps=args.steps,
                           seed=args.seed, constraint_aware=True)
    agent = DQNAgent.load(checkpoint, seed=args.seed)
    obs = env.reset(seed=args.seed)
    episode_return = 0.
    totals = {key: 0. for key in ("memory_penalty", "restoration_cost", "lost_work_cost",
                                  "task_disruption_cost", "action_cost", "deferred_event_cost")}
    revisit_events = useful = 0
    peak_memory = env.initial_memory
    print(f"TABasco demo | model seed={args.model_seed} | episode seed={args.seed} | "
          f"checkpoint={checkpoint}")
    print(f"initial active memory={env.initial_memory:.2f}/{env.memory_budget:.2f}; "
          f"observation shape={env.observation_space_shape}")

    while True:
        mask = env.action_mask(obs)
        action = agent.act(obs, explore=False, action_mask=mask)
        action_text = describe_action(action, env)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_return += reward
        peak_memory = max(peak_memory, info["active_memory"])
        for key in totals:
            totals[key] += info["reward_components"][key]
        revisit_events += info["revisit_event_count"]
        useful += info["useful_revisit_count"]
        event_names = {
            "active_revisit": "active tab revisited",
            "suspended_revisit": "suspended tab restored automatically",
            "suspended_revisit_deferred": "suspended revisit deferred",
            "closed_tab_recovered": "closed tab recreated",
            "closed_recovery_deferred": "closed-tab recovery deferred",
        }
        automatic = ", ".join(event_names.get(name, name.replace("_", " "))
                              for name in info["revisit_events"]) or "none"
        if info["arrival_created"]:
            automatic += ", new tab arrived"
        print(f"t={env.t:03d} {action_text:<34} reward={reward:7.3f} "
              f"memory={info['active_memory']:.2f}/{env.memory_budget:.2f} "
              f"valid={info['action_valid']} auto={automatic}")
        if terminated or truncated:
            break

    print("\nEpisode summary")
    print(f"return={episode_return:.3f}; steps={env.t}; memory violations={env.memory_violations}; "
          f"peak active memory={peak_memory:.2f}/{env.memory_budget:.2f}")
    print(f"useful revisit events={useful}/{revisit_events} "
          f"({useful / revisit_events:.3f} success rate)" if revisit_events else
          "useful revisit events=0/0 (no revisit events)")
    print("cost totals=" + ", ".join(f"{k}={v:.3f}" for k, v in totals.items()))
    print(f"reward accounting check: return + costs = {episode_return + sum(totals.values()):.3g}")


if __name__ == "__main__":
    main()
