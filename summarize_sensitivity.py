"""Create validation sweep, historical reward decomposition, and event audits."""
import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "sensitivity"


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1)), float(1.96 * values.std(ddof=1) / np.sqrt(len(values)))


def main():
    heldout_rows = []
    for setting, coefficient in (("soft_m1", 1), ("soft_m3", 3), ("soft_m5", 5),
                                ("constraint_aware", 1)):
        directory = "heldout_constraint" if setting == "constraint_aware" else f"heldout_m{coefficient}"
        for row in read_rows(OUT / directory / "evaluation_summary.csv"):
            record = {"setting": setting, "memory_coefficient": coefficient,
                      "constraint_aware": int(setting == "constraint_aware"),
                      "policy": row["policy"], "episodes": row["episodes"]}
            for metric in ("return", "violation_steps", "violation_step_fraction", "mean_excess_per_step",
                           "mean_excess_when_violating", "peak_memory_excess",
                           "restoration_recovery_cost", "lost_work_cost",
                           "useful_revisit_success_rate", "deferred_event_cost",
                           "memory_penalty_cost", "task_disruption_cost"):
                for suffix in ("mean", "std", "ci95_low", "ci95_high"):
                    key = f"{metric}_{suffix}"
                    record[key] = row.get(key, "")
            heldout_rows.append(record)
    write_rows(OUT / "heldout_all_variants_summary.csv", heldout_rows)

    sensitivity = []
    for coefficient in (1, 3, 5):
        raw = read_rows(OUT / f"validation_m{coefficient}" / "evaluation_episodes.csv")
        dqn = [row for row in raw if row["policy"].startswith("DQN_")]
        for seed_label in ("DQN_seed3090", "DQN_seed4090"):
            episodes = [row for row in dqn if row["policy"] == seed_label]
            row = {"coefficient": coefficient, "policy": seed_label,
                   "episodes": len(episodes)}
            for metric in ("return", "violation_step_fraction", "mean_excess_per_step",
                           "mean_excess_when_violating", "peak_memory_excess",
                           "restoration_recovery_cost", "lost_work_cost",
                           "useful_revisit_success_rate"):
                mean, sd, half = summarize([float(x[metric]) for x in episodes])
                row.update({f"{metric}_mean": mean, f"{metric}_std": sd,
                            f"{metric}_ci95_low": mean - half,
                            f"{metric}_ci95_high": mean + half})
            sensitivity.append(row)
    write_rows(OUT / "validation_sensitivity_summary.csv", sensitivity)

    # Historical corrected-run component audit; kept separate from new experiments.
    previous = read_rows(ROOT / "results" / "corrected" / "evaluation" /
                         "evaluation_episodes.csv")
    policies = sorted({row["policy"] for row in previous})
    reward_rows = []
    fields = {"memory": "memory_penalty_cost", "restoration": "restoration_recovery_cost",
              "lost_work": "lost_work_cost", "disruption": "task_disruption_cost",
              "action": "action_cost"}
    for policy in policies:
        episodes = [row for row in previous if row["policy"] == policy]
        result = {"policy": policy, "episodes": len(episodes)}
        for label, key in fields.items():
            result[f"{label}_cost_mean"] = float(np.mean([float(x[key]) for x in episodes]))
        result["return_mean"] = float(np.mean([float(x["return"]) for x in episodes]))
        result["violation_steps_mean"] = float(np.mean([float(x["violation_steps"]) for x in episodes]))
        result["violation_step_fraction_mean"] = result["violation_steps_mean"] / 120.
        reward_rows.append(result)
    write_rows(OUT / "previous_corrected_reward_decomposition.csv", reward_rows)

    # Source labels are the persistent first-crossing cause for each violating
    # post-transition step; crossing counts below are instantaneous stage changes.
    diagnostic = read_rows(OUT / "diagnostics" / "evaluation_episodes.csv")
    source_metrics = ["action_crossing_steps", "suspended_revisit_crossing_steps",
                      "closed_recovery_crossing_steps", "arrival_crossing_steps",
                      "violation_steps_initial_state", "violation_steps_agent_action_source",
                      "violation_steps_suspended_revisit_source",
                      "violation_steps_closed_recovery_source",
                      "violation_steps_new_arrival_source"]
    event_rows = []
    for policy in sorted({row["policy"] for row in diagnostic}):
        subset = [row for row in diagnostic if row["policy"] == policy]
        result = {"policy": policy, "episodes": len(subset)}
        for metric in source_metrics:
            result[f"{metric}_mean"] = float(np.mean([float(row[metric]) for row in subset]))
        event_rows.append(result)
    write_rows(OUT / "validation_event_attribution.csv", event_rows)
    print("Wrote sensitivity summary, historical reward decomposition, and validation event attribution.")


if __name__ == "__main__":
    main()
