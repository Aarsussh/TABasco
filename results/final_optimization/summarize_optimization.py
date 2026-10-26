"""Build final-optimization CSVs and plots from the recorded episode data."""
from __future__ import annotations

import csv
import math
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "final_optimization"


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        fieldnames = list(dict.fromkeys(key for row in data for key in row))
        w = csv.DictWriter(stream, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(data)


def summary(values):
    v = np.asarray(values, dtype=float)
    mean = float(v.mean())
    sd = float(v.std(ddof=1)) if len(v) > 1 else 0.
    half = 1.96 * sd / math.sqrt(len(v))
    return mean, sd, mean - half, mean + half


def main():
    configs = ["control"]
    dirs = {"control": OUT / "control_validation"}
    for name in ("lost_1p5", "restore_0p6", "disrupt_0p35", "balanced"):
        configs.append(name)
        dirs[name] = OUT / f"screen_{name}"
    for name in ("lost_1p5", "disrupt_0p35"):
        key = f"{name}_two_seed"
        configs.append(key)
        dirs[key] = OUT / f"confirm_{name}"

    metric_map = {
        "canonical_return": "canonical_return" if "canonical_return" in rows(dirs["lost_1p5"] / "evaluation_episodes.csv")[0] else "return",
        "policy_return": "return",
        "useful_revisit_success_rate": "useful_revisit_success_rate",
        "restoration_cost": "restoration_cost" if "restoration_cost" in rows(dirs["lost_1p5"] / "evaluation_episodes.csv")[0] else "restoration_recovery_cost",
        "lost_work_cost": "lost_work_cost",
        "disruption_cost": "disruption_cost" if "disruption_cost" in rows(dirs["lost_1p5"] / "evaluation_episodes.csv")[0] else "task_disruption_cost",
        "deferred_cost": "deferred_cost" if "deferred_cost" in rows(dirs["lost_1p5"] / "evaluation_episodes.csv")[0] else "deferred_event_cost",
        "invalid_action_rate": "invalid_action_rate",
        "violation_steps": "violation_steps",
        "peak_active_memory": "peak_active_memory",
        "reward_accounting_error": "reward_accounting_error",
    }
    output = []
    for config in configs:
        if config.endswith("_two_seed"):
            variant = config.removesuffix("_two_seed")
            evpath = dirs[config] / "evaluation_episodes.csv"
        elif config == "control":
            evpath = dirs[config] / "evaluation_episodes.csv"
            variant = config
        else:
            evpath = dirs[config] / "evaluation_episodes.csv"
            variant = config
        source = rows(evpath)
        for policy in sorted(set(r["policy"] for r in source)):
            subset = [r for r in source if r["policy"] == policy]
            rec = {"configuration": config, "policy": policy,
                   "episodes": len(subset), "environment_seeds": "92000-92039"}
            for metric, field in metric_map.items():
                aliases = {"canonical_return": "return",
                           "restoration_cost": "restoration_recovery_cost",
                           "disruption_cost": "task_disruption_cost",
                           "deferred_cost": "deferred_event_cost"}
                if field not in subset[0] and metric in aliases:
                    field = aliases[metric]
                vals = [float(r[field]) for r in subset]
                m, sd, lo, hi = summary(vals)
                for suffix, value in zip(("mean", "std", "ci95_low", "ci95_high"),
                                         (m, sd, lo, hi)):
                    rec[f"{metric}_{suffix}"] = value
            output.append(rec)
    write(OUT / "validation_summary.csv", output)

    # Paired canonical-reward comparison between each candidate and the
    # current model trained with the matching seed. Screening rows use seed 3090.
    control = rows(dirs["control"] / "evaluation_episodes.csv")
    control_ret = {(int(r["seed"]), r["policy"]): float(r["return"]) for r in control}
    paired = []
    for config in ("lost_1p5", "restore_0p6", "disrupt_0p35", "balanced",
                   "lost_1p5_two_seed", "disrupt_0p35_two_seed"):
        data = rows(dirs[config] / "evaluation_episodes.csv")
        for r in data:
            if not r["policy"].startswith("DQN"):
                continue
            training_seed = r["policy"].split("_")[-1]
            control_label = f"DQN_seed{training_seed}"
            candidate_return = float(r.get("canonical_return", r["return"]))
            paired.append({"configuration": config, "training_seed": training_seed,
                           "environment_seed": int(r["seed"]), "policy": r["policy"],
                           "control_policy": control_label,
                           "candidate_canonical_return": candidate_return,
                           "control_return": control_ret[(int(r["seed"]), control_label)],
                           "candidate_minus_control": candidate_return -
                           control_ret[(int(r["seed"]), control_label)]})
    write(OUT / "validation_paired_candidate_control_episodes.csv", paired)
    comparisons = []
    for config in ("lost_1p5", "restore_0p6", "disrupt_0p35", "balanced",
                   "lost_1p5_two_seed", "disrupt_0p35_two_seed"):
        subset = [r for r in paired if r["configuration"] == config]
        record = {"configuration": config,
                  "evaluation_seeds": "92000-92039",
                  "training_seed_count": len(set(r["training_seed"] for r in subset))}
        seed_means = []
        for seed in sorted(set(r["training_seed"] for r in subset)):
            seed_values = [r["candidate_minus_control"] for r in subset
                           if r["training_seed"] == seed]
            m, sd, lo, hi = summary(seed_values)
            seed_means.append(m)
            record.update({f"seed{seed}_paired_episodes": len(seed_values),
                           f"seed{seed}_mean_difference": m,
                           f"seed{seed}_difference_sd": sd,
                           f"seed{seed}_ci95_low": lo,
                           f"seed{seed}_ci95_high": hi})
        record["mean_of_training_seed_effects"] = float(np.mean(seed_means))
        record["training_seed_effect_sd"] = (float(np.std(seed_means, ddof=1))
                                               if len(seed_means) > 1 else "")
        record["uncertainty_note"] = (
            "Per-seed intervals condition on each trained model; mean of two seed effects is descriptive, not a training-population CI."
            if len(seed_means) > 1 else
            "Paired interval is conditional on the single screening model; no training-seed uncertainty estimate.")
        comparisons.append(record)
    write(OUT / "validation_paired_candidate_control_summary.csv", comparisons)

    # Preserve the already completed prior held-out evaluation as a clearly
    # identified reused artifact. Never rerun it for this selection decision.
    heldout_dir = OUT / "heldout_existing"
    heldout_dir.mkdir(parents=True, exist_ok=True)
    source_dir = ROOT / "results" / "sensitivity" / "heldout_constraint"
    for filename in ("evaluation_episodes.csv", "evaluation_summary.csv",
                     "paired_return_differences.csv", "evaluation_config.json"):
        shutil.copy2(source_dir / filename, heldout_dir / filename)
    (heldout_dir / "PROVENANCE.txt").write_text(
        "Copied without rerunning from results/sensitivity/heldout_constraint. "
        "This held-out block (seeds 91000-91059) had already been inspected in "
        "the previous TABasco experiment. It was not used for candidate selection.\n")
    held_summary = rows(heldout_dir / "evaluation_summary.csv")
    decomp_fields = ("return_mean", "return_std", "return_ci95_low", "return_ci95_high",
                     "violation_steps_mean", "violation_step_fraction_mean",
                     "mean_excess_per_step_mean", "peak_memory_excess_mean",
                     "useful_revisit_success_rate_mean", "restoration_recovery_cost_mean",
                     "lost_work_cost_mean", "task_disruption_cost_mean",
                     "deferred_event_cost_mean", "action_cost_mean",
                     "memory_penalty_cost_mean", "invalid_action_rate_mean")
    decomp = [{"policy": r["policy"], "episodes": r["episodes"],
               **{field: r[field] for field in decomp_fields}}
              for r in held_summary]
    write(OUT / "heldout_reward_decomposition.csv", decomp)
    print(f"Wrote {OUT / 'validation_summary.csv'} and paired comparison files.")
    print(f"Reused prior held-out results in {heldout_dir} without reevaluation.")


if __name__ == "__main__":
    main()
