"""Plot validation reward-coefficient sensitivity from recorded CSVs."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "results" / "sensitivity"
OUT = BASE / "plots"


def read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = read(BASE / "validation_sensitivity_summary.csv")
    coefficients = [1, 3, 5]
    policies = ["DQN_seed3090", "DQN_seed4090"]
    styles = {"DQN_seed3090": "o-", "DQN_seed4090": "s--"}
    for metric, ylabel, title, filename in (
        ("return", "Validation episodic return", "Reward coefficient sensitivity: return", "sensitivity_return.png"),
        ("violation_step_fraction", "Fraction of steps over budget", "Reward coefficient sensitivity: violation frequency", "sensitivity_violation_fraction.png"),
        ("mean_excess_per_step", "Mean excess memory per step (synthetic units)", "Reward coefficient sensitivity: excess magnitude", "sensitivity_excess.png"),
    ):
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for policy in policies:
            subset = [next(row for row in rows if int(row["coefficient"]) == c and row["policy"] == policy)
                      for c in coefficients]
            means = np.array([float(row[f"{metric}_mean"]) for row in subset])
            low = np.array([float(row[f"{metric}_ci95_low"]) for row in subset])
            high = np.array([float(row[f"{metric}_ci95_high"]) for row in subset])
            ax.errorbar(coefficients, means, yerr=np.vstack([means - low, high - means]),
                        fmt=styles[policy], capsize=4, linewidth=1.8,
                        label=policy.replace("DQN_seed", "training seed "))
        ax.set(xlabel="Memory-penalty coefficient", ylabel=ylabel, title=title,
               xticks=coefficients)
        ax.grid(alpha=.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / filename, dpi=190)
        plt.close(fig)
    print(f"Validation sensitivity plots saved under {OUT}")


if __name__ == "__main__":
    main()
