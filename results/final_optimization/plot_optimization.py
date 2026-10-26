"""Plots generated only from final_optimization CSV evaluation outputs."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "final_optimization"
PLOTS = OUT / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.titlesize": 12,
                     "axes.labelsize": 10, "figure.dpi": 140})


def read(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(PLOTS / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    summary = read(OUT / "validation_summary.csv")
    chosen = [r for r in summary if r["policy"].startswith("DQN")]
    # Show all candidate means using canonical control reward. Screening and
    # confirmatory models are separated visibly in labels.
    labels, means, errors = [], [], []
    for name in ("control", "lost_1p5", "restore_0p6", "disrupt_0p35",
                 "balanced", "lost_1p5_two_seed", "disrupt_0p35_two_seed"):
        for r in chosen:
            if r["configuration"] != name:
                continue
            labels.append(f"{name}\n{r['policy'].replace('DQN_', '')}")
            means.append(float(r["canonical_return_mean"]))
            errors.append((float(r["canonical_return_mean"]) - float(r["canonical_return_ci95_low"]),
                           float(r["canonical_return_ci95_high"]) - float(r["canonical_return_mean"])))
    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = np.arange(len(labels))
    yerr = np.array(errors).T
    colors = ["#38598c" if "control" in label else
              "#d17b0f" if "two_seed" in label else "#70a37f" for label in labels]
    ax.bar(x, means, color=colors, alpha=.9)
    ax.errorbar(x, means, yerr=yerr, fmt="none", ecolor="#222", capsize=3)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("Mean episodic return under control reward")
    ax.set_title("Validation return: common original reward, seeds 92000–92039")
    ax.grid(axis="y", alpha=.25)
    save(fig, "validation_canonical_return.png")

    held = read(OUT / "heldout_existing" / "evaluation_summary.csv")
    labels = [r["policy"].replace("DQN_seed", "DQN ") for r in held]
    ret = [float(r["return_mean"]) for r in held]
    err = [float(r["return_std"]) * 1.96 / np.sqrt(int(r["episodes"])) for r in held]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = np.arange(len(labels))
    ax.bar(x, ret, color=["#38598c" if s.startswith("DQN") else "#999" for s in labels])
    ax.errorbar(x, ret, yerr=err, fmt="none", ecolor="#222", capsize=4)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean episodic return (95% CI)")
    ax.set_title("Previously completed held-out comparison (seeds 91000–91059)")
    ax.grid(axis="y", alpha=.25)
    save(fig, "heldout_return_comparison.png")

    useful = [float(r["useful_revisit_success_rate_mean"]) for r in held]
    fig, ax = plt.subplots(figsize=(8.5, 4.3))
    ax.bar(x, useful, color=["#38598c" if s.startswith("DQN") else "#999" for s in labels])
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Useful revisit success rate")
    ax.set_title("Held-out useful-revisit success")
    ax.grid(axis="y", alpha=.25)
    save(fig, "heldout_useful_revisit.png")

    components = ["restoration_recovery_cost", "lost_work_cost",
                  "task_disruption_cost", "deferred_event_cost",
                  "action_cost", "memory_penalty_cost"]
    component_labels = ["Restoration", "Lost work", "Disruption",
                        "Deferred events", "Action", "Memory penalty"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    width = .13
    for j, field in enumerate(components):
        vals = [float(r[f"{field}_mean"]) for r in held]
        ax.bar(x + (j - 2.5) * width, vals, width, label=component_labels[j])
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean cost per episode")
    ax.set_title("Held-out reward-cost decomposition")
    ax.legend(ncol=3)
    ax.grid(axis="y", alpha=.25)
    save(fig, "heldout_reward_components.png")
    print(f"Saved four plots under {PLOTS}")


if __name__ == "__main__":
    main()
