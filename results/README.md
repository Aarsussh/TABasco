# Experiment map

This folder contains result artifacts, not source code. Keep historical and selected experiments separate; CSVs preserve per-episode measurements and JSON files preserve run configuration.

| Directory | Role | Status |
|---|---|---|
| `v1/` | First implementation and evaluation | Historical; superseded |
| `corrected/` | Corrected soft-budget runs | Historical comparison; memory is not guaranteed |
| `sensitivity/` | Memory penalty sweep, constraint-aware validation, and saved held-out evaluation | Selected constrained model results; held-out seeds 91000–91059 were previously examined |
| `final_optimization/` | Reward-weight candidates screened on validation | Follow-up; no candidate replaced selected checkpoints |
| `reproducibility_check/` | Five-episode CLI smoke evaluation on seeds 93000-93004, with plots | Verification only; not used for report or selection |
| Root CSV/PNG files | Original project run outputs | Preserved as historical outputs; use their adjacent configs or the report for provenance |

Primary reproducibility targets are `sensitivity/heldout_constraint/evaluation_episodes.csv`, `evaluation_summary.csv`, and `paired_return_differences.csv`. Selected checkpoints are `../models/sensitivity/constrained_seed3090.pt` and `constrained_seed4090.pt`. The complete experiment drivers are in the repository root or, for the reward-weight follow-up, inside `final_optimization/`.

The selected held-out evaluation is a previously examined common-seed block, not an untouched test. Do not use it for additional model or coefficient selection. See the repository README and `../report/report.tex` for measured outcomes and interpretation.
