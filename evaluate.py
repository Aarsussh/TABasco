"""Common-seed held-out evaluation and paired confidence intervals."""
import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np
import torch

from agent import DQNAgent
from baselines import lfu_action, lru_action, random_valid_action
from environment import TabManagementEnv


def parse_checkpoints(values):
    if not values:
        return [("DQN", "models/tabasco_dqn.pt")]
    parsed = []
    for value in values:
        if "=" in value:
            label, path = value.split("=", 1)
            parsed.append((f"DQN_seed{label}", path))
        else:
            parsed.append(("DQN", value))
    return parsed


def stats(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if not arr.size:
        return (float("nan"),) * 4
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if arr.size > 1 else 0.
    half = 1.96 * sd / math.sqrt(arr.size)
    return mean, sd, mean - half, mean + half


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", default=None,
                        help="checkpoint path, repeatable; use seed=path to label training seed")
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--seed", type=int, default=90000)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--memory-budget", type=float, default=8.)
    parser.add_argument("--memory-coefficient", type=float, default=1.)
    parser.add_argument("--constraint-aware", action="store_true")
    parser.add_argument("--n-tabs", type=int, default=12)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    checkpoints = parse_checkpoints(args.checkpoint)
    torch.set_num_threads(1)
    agents = [(label, DQNAgent.load(path, seed=args.seed)) for label, path in checkpoints]
    policies = {label: (lambda env, rng, model=agent:
                        model.act(env._obs(), explore=False, action_mask=env.action_mask()))
                for label, agent in agents}
    policies.update({"LRU": lambda env, rng: lru_action(env),
                     "LFU": lambda env, rng: lfu_action(env),
                     "Random": lambda env, rng: random_valid_action(env, rng)})
    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for episode in range(args.episodes):
        seed = args.seed + episode
        for policy, choose in policies.items():
            env = TabManagementEnv(args.n_tabs, args.memory_budget, args.max_steps, seed,
                                   coefficients={"memory": args.memory_coefficient},
                                   constraint_aware=args.constraint_aware)
            rng = random.Random(seed + 777)
            episode_return = 0.
            memory = []
            violation_steps = 0
            violation_magnitude = 0.
            peak_excess = 0.
            action_crossings = revisit_crossings = recovery_crossings = arrival_crossings = 0
            source_step_counts = {"initial_state": 0, "agent_action": 0,
                                  "suspended_revisit": 0, "closed_recovery": 0,
                                  "new_arrival": 0, "expiration": 0}
            arrivals = suspended_restores = closed_recoveries = 0
            restoration = lost_work = 0.
            memory_cost = disruption_cost = action_cost = deferred_cost = 0.
            component_sum = reward_accounting_error = 0.
            useful_revisits = revisit_count = invalid = 0
            initial_memory = env.initial_memory
            for _ in range(args.max_steps):
                action = choose(env, rng)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_return += reward
                memory.append(info["active_memory"])
                violation_steps += int(info["memory_violation"])
                violation_magnitude += info["memory_excess"]
                peak_excess = max(peak_excess, info["memory_excess"])
                crossing = info["memory_violation_crossing_source"]
                action_crossings += int(crossing == "agent_action")
                revisit_crossings += int(crossing == "suspended_revisit")
                recovery_crossings += int(crossing == "closed_recovery")
                arrival_crossings += int(crossing == "new_arrival")
                source = info["memory_violation_source"]
                if info["memory_violation"] and source in source_step_counts:
                    source_step_counts[source] += 1
                arrivals += int(info["arrival_created"])
                suspended_restores += int(info["automatic_revisit_restored"])
                closed_recoveries += int(info["closed_recovery_created"])
                restoration += info["restoration_cost"]
                lost_work += info["lost_work_cost"]
                memory_cost += info["memory_penalty"]
                disruption_cost += info["task_disruption_cost"]
                action_cost += info["action_cost"]
                deferred_cost += info["deferred_event_cost"]
                step_cost_sum = sum(info["reward_components"].values())
                component_sum += step_cost_sum
                reward_accounting_error += abs(reward + step_cost_sum)
                revisit_count += info["revisit_event_count"]
                useful_revisits += info["useful_revisit_count"]
                invalid += int(not info["action_valid"])
                if terminated or truncated:
                    break
            rows.append({"episode": episode + 1, "seed": seed, "policy": policy,
                         "return": episode_return, "mean_active_memory": float(np.mean(memory)),
                         "initial_active_memory": initial_memory,
                         "initial_memory_excess": max(0., initial_memory - args.memory_budget),
                         "peak_active_memory": max(memory, default=0.),
                         "violation_steps": violation_steps,
                         "violation_step_fraction": violation_steps / max(1, len(memory)),
                         "violation_magnitude": violation_magnitude,
                         "mean_excess_per_step": violation_magnitude / max(1, len(memory)),
                         "mean_excess_when_violating": violation_magnitude / violation_steps if violation_steps else 0.,
                         "peak_memory_excess": peak_excess,
                         "action_crossing_steps": action_crossings,
                         "suspended_revisit_crossing_steps": revisit_crossings,
                         "closed_recovery_crossing_steps": recovery_crossings,
                         "arrival_crossing_steps": arrival_crossings,
                         "violation_steps_initial_state": source_step_counts["initial_state"],
                         "violation_steps_agent_action_source": source_step_counts["agent_action"],
                         "violation_steps_suspended_revisit_source": source_step_counts["suspended_revisit"],
                         "violation_steps_closed_recovery_source": source_step_counts["closed_recovery"],
                         "violation_steps_new_arrival_source": source_step_counts["new_arrival"],
                         "violation_steps_expiration_source": source_step_counts["expiration"],
                         "automatic_new_arrivals": arrivals,
                         "automatic_suspended_restores": suspended_restores,
                         "automatic_closed_recoveries": closed_recoveries,
                         "restoration_recovery_cost": restoration,
                         "memory_penalty_cost": memory_cost,
                         "task_disruption_cost": disruption_cost,
                         "action_cost": action_cost,
                         "deferred_event_cost": deferred_cost,
                         "reward_component_cost_sum": component_sum,
                         "reward_accounting_error": reward_accounting_error,
                         "lost_work_cost": lost_work,
                         "revisit_events": revisit_count,
                         "useful_revisit_events": useful_revisits,
                         "useful_revisit_success_rate": useful_revisits / revisit_count if revisit_count else float("nan"),
                         "invalid_actions": invalid,
                         "invalid_action_rate": invalid / max(1, len(memory))})

    fields = list(rows[0].keys())
    with (out / "evaluation_episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    metric_names = [field for field in fields if field not in ("episode", "seed", "policy")]
    summary_fields = ["policy", "episodes"] + [f"{name}_{suffix}" for name in metric_names
                      for suffix in ("mean", "std", "ci95_low", "ci95_high")]
    summary = []
    for policy in policies:
        subset = [row for row in rows if row["policy"] == policy]
        result = {"policy": policy, "episodes": len(subset)}
        for name in metric_names:
            mean, sd, low, high = stats([row[name] for row in subset])
            result.update({f"{name}_mean": mean, f"{name}_std": sd,
                           f"{name}_ci95_low": low, f"{name}_ci95_high": high})
        summary.append(result)
    with (out / "evaluation_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary)

    # Paired return differences use the same held-out session seed for both policies.
    pivots = {(row["seed"], row["policy"]): row["return"] for row in rows}
    paired = []
    dqn_policies = [label for label, _ in agents]
    for dqn_policy in dqn_policies:
        for baseline in ("LRU", "LFU", "Random"):
            diffs = [pivots[(args.seed + ep, dqn_policy)] - pivots[(args.seed + ep, baseline)]
                     for ep in range(args.episodes)]
            mean, sd, low, high = stats(diffs)
            paired.append({"dqn_policy": dqn_policy, "baseline": baseline,
                           "episodes": len(diffs), "dqn_minus_baseline_mean": mean,
                           "difference_sd": sd, "ci95_low": low, "ci95_high": high})
    with (out / "paired_return_differences.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=paired[0].keys())
        writer.writeheader()
        writer.writerows(paired)
    with (out / "evaluation_config.json").open("w") as stream:
        import json
        config_env = TabManagementEnv(args.n_tabs, args.memory_budget, args.max_steps, args.seed,
                                      coefficients={"memory": args.memory_coefficient},
                                      constraint_aware=args.constraint_aware)
        json.dump({**vars(args), "checkpoints": checkpoints,
                   "evaluation_seeds": [args.seed, args.seed + args.episodes - 1],
                   "environment_coefficients": config_env.c,
                   "constraint_aware": config_env.constraint_aware,
                   "constraint_event_rule": ("blocked/deferred arrivals; suspended and closed returns remain pending when restore does not fit; initial overflow tabs start suspended" if args.constraint_aware else "soft memory budget; automatic events may exceed it"),
                   "baseline_candidate_set": "active-tab SUSPEND actions; WAIT if empty",
                   "useful_revisit_success": "useful task-matching revisit events / all revisit events",
                   "invalid_action_rate": "invalid decisions / environment steps",
                   "ci_method": "normal-approximation 95% CI over episodes; paired differences by common seed"},
                  stream, indent=2)
    print((out / "evaluation_summary.csv").read_text())
    print((out / "paired_return_differences.csv").read_text())


if __name__ == "__main__":
    main()
