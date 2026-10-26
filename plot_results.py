"""Generate corrected-run plots only from the saved CSV outputs."""
import argparse
import csv
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "tabasco-mpl-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path):
    try:
        with Path(path).open(newline="") as stream:
            return list(csv.DictReader(stream))
    except (FileNotFoundError, OSError):
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/corrected")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--memory-budget", type=float, default=8.)
    args = parser.parse_args()
    root = Path(args.results_dir)
    out = Path(args.output_dir) if args.output_dir else root / "plots"
    out.mkdir(parents=True, exist_ok=True)

    training_files = sorted(root.glob("train_seed*/training.csv"))
    if not training_files:
        one_training = root / "training.csv"
        if one_training.exists():
            training_files = [one_training]
    if training_files:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for path in training_files:
            rows = read_csv(path)
            if not rows:
                continue
            x = np.array([int(row["episode"]) for row in rows])
            y = np.array([float(row["return"]) for row in rows])
            window = min(20, max(1, len(y) // 10))
            rolling = np.convolve(y, np.ones(window) / window, mode="valid")
            label = path.parent.name.replace("train_", "")
            ax.plot(x, y, alpha=.18, linewidth=.7)
            ax.plot(x[window - 1:], rolling, linewidth=2, label=f"{label}: {window}-episode mean")
        ax.set(xlabel="Training episode", ylabel="Episode return",
               title="Corrected DQN training (thin: episode; solid: rolling mean)")
        ax.grid(alpha=.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "training_returns.png", dpi=180)
        plt.close(fig)

    evaluation_root = root / "evaluation" if (root / "evaluation" / "evaluation_summary.csv").exists() else root
    summary = read_csv(evaluation_root / "evaluation_summary.csv")
    policies = [row["policy"] for row in summary]
    if summary:
        x = np.arange(len(summary))
        means = np.array([float(row["return_mean"]) for row in summary])
        lows = np.array([float(row["return_ci95_low"]) for row in summary])
        highs = np.array([float(row["return_ci95_high"]) for row in summary])
        errors = np.vstack([means - lows, highs - means])
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(x, means, yerr=errors, capsize=4)
        ax.set_xticks(x, policies, rotation=15)
        ax.set(ylabel="Mean held-out return", title="Common-seed policy comparison (95% CI)")
        ax.grid(axis="y", alpha=.25)
        fig.tight_layout()
        fig.savefig(out / "return_comparison.png", dpi=180)
        plt.close(fig)

        memory = np.array([float(row["mean_active_memory_mean"]) for row in summary])
        memlo = np.array([float(row["mean_active_memory_ci95_low"]) for row in summary])
        memhi = np.array([float(row["mean_active_memory_ci95_high"]) for row in summary])
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(x, memory, yerr=np.vstack([memory - memlo, memhi - memory]), capsize=4)
        ax.axhline(args.memory_budget, color="red", linestyle="--", label="Memory budget")
        ax.set_xticks(x, policies, rotation=15)
        ax.set(ylabel="Mean active memory (synthetic units)", title="Active memory by policy (95% CI)")
        ax.legend()
        ax.grid(axis="y", alpha=.25)
        fig.tight_layout()
        fig.savefig(out / "memory_comparison.png", dpi=180)
        plt.close(fig)

        violations = np.array([float(row["violation_steps_mean"]) for row in summary])
        vlo = np.array([float(row["violation_steps_ci95_low"]) for row in summary])
        vhi = np.array([float(row["violation_steps_ci95_high"]) for row in summary])
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(x, violations, yerr=np.vstack([violations - vlo, vhi - violations]), capsize=4)
        ax.set_xticks(x, policies, rotation=15)
        ax.set(ylabel="Violation steps per episode", title="Memory-budget violations (95% CI)")
        ax.grid(axis="y", alpha=.25)
        fig.tight_layout()
        fig.savefig(out / "violation_comparison.png", dpi=180)
        plt.close(fig)
    else:
        print(f"No evaluation_summary.csv found under {root}; evaluation plots skipped.")
    if not training_files:
        print(f"No corrected training CSV found under {root}; training plot skipped.")
    print(f"Plots saved under {out}")


if __name__ == "__main__":
    main()
