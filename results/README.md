# Experiment map

This folder contains experiment artifacts. CSVs preserve per-episode measurements, JSON files preserve run configurations, and PNGs contain plots generated from those results. Historical and selected experiments remain separated for traceability.

| Directory | Role | Status |
|---|---|---|
| `v1/` | First implementation and evaluation | Historical; superseded |
| `corrected/` | Corrected soft-budget runs | Historical comparison; memory can exceed the budget |
| `sensitivity/` | Memory-penalty sweep, constraint-aware validation, and selected evaluation | Selected constrained-model results; seeds 91000–91059 were previously examined |
| `final_optimization/` | Reward-weight candidates screened on validation | Follow-up; no candidate replaced the selected checkpoints |
| `reproducibility_check/` | Five-episode CLI smoke evaluation on seeds 93000–93004, with plots | Verification only; not used for selection |
| Root CSV/PNG files | Original project run outputs | Historical artifacts; use adjacent configurations and experiment notes for context |

Primary reproduction data are `sensitivity/heldout_constraint/evaluation_episodes.csv`, `evaluation_summary.csv`, and `paired_return_differences.csv`. Selected checkpoints are `../models/sensitivity/constrained_seed3090.pt` and `../models/sensitivity/constrained_seed4090.pt`. Experiment drivers are in the repository root or, for reward-weight follow-up, in `final_optimization/`.

The selected held-out evaluation is a previously examined common-seed block, not an untouched test set. Do not use it for further model or coefficient selection. See the root README and `sensitivity/heldout_constraint/` for measured outcomes and raw data.
