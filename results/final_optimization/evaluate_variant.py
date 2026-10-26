"""Evaluate a reward variant on common seeds using unchanged TABasco policies."""
from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agent import DQNAgent  # noqa: E402
from baselines import lfu_action, lru_action, random_valid_action  # noqa: E402
from environment import TabManagementEnv  # noqa: E402


def stats(values):
    a = np.asarray(values, dtype=float)
    m = float(np.mean(a))
    s = float(np.std(a, ddof=1)) if len(a) > 1 else 0.
    h = 1.96 * s / math.sqrt(len(a))
    return m, s, m - h, m + h


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--seed-start", type=int, default=92000)
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--lost", type=float, default=1.2)
    p.add_argument("--restore", type=float, default=.8)
    p.add_argument("--disruption", type=float, default=.45)
    p.add_argument("--checkpoint", action="append", default=[],
                   help="label=path; repeat for each model")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    models = []
    for item in a.checkpoint:
        label, path = item.split("=", 1)
        models.append((f"DQN_{label}", DQNAgent.load(ROOT / path, seed=a.seed_start)))
    policies = {label: (lambda env, rng, model=model:
                        model.act(env._obs(), explore=False,
                                  action_mask=env.action_mask()))
                for label, model in models}
    policies.update({"LRU": lambda env, rng: lru_action(env),
                     "LFU": lambda env, rng: lfu_action(env),
                     "Random": lambda env, rng: random_valid_action(env, rng)})
    coeff = {"memory": 1., "restore": a.restore, "lost": a.lost,
             "disruption": a.disruption, "action": .035,
             "invalid": .12, "deferred": .25}
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    metric_fields = ["return", "canonical_return", "mean_active_memory", "peak_active_memory",
                     "violation_steps", "violation_step_fraction", "violation_magnitude",
                     "mean_excess_per_step", "peak_memory_excess", "restoration_cost",
                     "lost_work_cost", "disruption_cost", "deferred_cost", "action_cost",
                     "memory_cost", "useful_revisit_success_rate", "revisit_events",
                     "useful_revisit_events", "invalid_action_rate", "episode_length",
                     "reward_accounting_error"]
    for ep in range(a.episodes):
        seed = a.seed_start + ep
        for policy, choose in policies.items():
            env = TabManagementEnv(n_tabs=12, memory_budget=8., max_steps=120,
                                   seed=seed, coefficients=coeff, constraint_aware=True)
            rng = random.Random(seed + 777)
            ret = 0.
            canonical_ret = 0.
            mem = []
            accum = {k: 0. for k in ("restoration", "lost_work", "disruption",
                                      "deferred", "action", "memory")}
            violation_steps = excess = peak_excess = 0.
            revisits = useful = invalid = accounting_error = 0.
            for step in range(120):
                action = choose(env, rng)
                _, reward, terminated, truncated, info = env.step(action)
                ret += reward
                # Score the exact same trajectory with the immutable control
                # reward weights so cross-configuration policy comparisons
                # remain on one scale. Dynamics do not depend on reward.
                canonical_ret -= (info["memory_penalty"] +
                                  .8 / max(coeff["restore"], 1e-12) * info["restoration_cost"] +
                                  1.2 / max(coeff["lost"], 1e-12) * info["lost_work_cost"] +
                                  .45 / max(coeff["disruption"], 1e-12) * info["task_disruption_cost"] +
                                  info["action_cost"] + info["deferred_event_cost"])
                mem.append(info["active_memory"])
                violation_steps += int(info["memory_violation"])
                excess += info["memory_excess"]
                peak_excess = max(peak_excess, info["memory_excess"])
                accum["restoration"] += info["restoration_cost"]
                accum["lost_work"] += info["lost_work_cost"]
                accum["disruption"] += info["task_disruption_cost"]
                accum["deferred"] += info["deferred_event_cost"]
                accum["action"] += info["action_cost"]
                accum["memory"] += info["memory_penalty"]
                revisits += info["revisit_event_count"]
                useful += info["useful_revisit_count"]
                invalid += int(not info["action_valid"])
                accounting_error += abs(reward + sum(info["reward_components"].values()))
                if env._active_memory() > env.memory_budget + 1e-7:
                    raise RuntimeError(f"constraint invariant failed: {policy} seed {seed}")
                if terminated or truncated:
                    break
            row = {"name": a.name, "episode": ep + 1, "seed": seed,
                   "policy": policy, "return": ret, "canonical_return": canonical_ret,
                   "mean_active_memory": float(np.mean(mem)),
                   "peak_active_memory": max(mem, default=0.),
                   "violation_steps": violation_steps,
                   "violation_step_fraction": violation_steps / max(1, len(mem)),
                   "violation_magnitude": excess,
                   "mean_excess_per_step": excess / max(1, len(mem)),
                   "peak_memory_excess": peak_excess,
                   "restoration_cost": accum["restoration"],
                   "lost_work_cost": accum["lost_work"],
                   "disruption_cost": accum["disruption"],
                   "deferred_cost": accum["deferred"],
                   "action_cost": accum["action"], "memory_cost": accum["memory"],
                   "useful_revisit_success_rate": useful / revisits if revisits else math.nan,
                   "revisit_events": revisits, "useful_revisit_events": useful,
                   "invalid_action_rate": invalid / max(1, len(mem)),
                   "episode_length": len(mem), "reward_accounting_error": accounting_error}
            rows.append(row)
    with (out / "evaluation_episodes.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    summary = []
    for policy in policies:
        subset = [r for r in rows if r["policy"] == policy]
        result = {"name": a.name, "policy": policy, "episodes": len(subset)}
        for metric in metric_fields:
            vals = [r[metric] for r in subset if math.isfinite(float(r[metric]))]
            m, s, lo, hi = stats(vals)
            for suffix, val in zip(("mean", "std", "ci95_low", "ci95_high"), (m, s, lo, hi)):
                result[f"{metric}_{suffix}"] = val
        summary.append(result)
    with (out / "evaluation_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader(); w.writerows(summary)
    lookup = {(r["seed"], r["policy"]): r["canonical_return"] for r in rows}
    paired = []
    for label, _ in models:
        for baseline in ("LRU", "LFU", "Random"):
            diffs = [lookup[(seed, label)] - lookup[(seed, baseline)]
                     for seed in range(a.seed_start, a.seed_start + a.episodes)]
            m, s, lo, hi = stats(diffs)
            paired.append({"name": a.name, "metric": "canonical_return",
                           "dqn_policy": label,
                           "baseline": baseline, "episodes": len(diffs),
                           "difference_mean": m, "difference_std": s,
                           "ci95_low": lo, "ci95_high": hi})
    with (out / "paired_return_differences.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(paired[0]))
        w.writeheader(); w.writerows(paired)
    import json
    (out / "evaluation_config.json").write_text(json.dumps({
        "name": a.name, "episodes": a.episodes,
        "evaluation_seeds": [a.seed_start, a.seed_start + a.episodes - 1],
        "coefficients": coeff, "constraint_aware": True,
        "checkpoints": a.checkpoint,
        "canonical_reward": {"memory": 1., "restore": .8, "lost": 1.2,
                             "disruption": .45, "action": .035, "deferred": .25},
        "selection_use": "validation only",
        "cross_candidate_primary_metric": "canonical_return on exact candidate trajectories"}, indent=2))
    print((out / "evaluation_summary.csv").read_text())


if __name__ == "__main__":
    main()
