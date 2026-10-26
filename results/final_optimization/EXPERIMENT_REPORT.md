# TABasco final reward-weight optimization

## Recommendation

Retain the existing constraint-aware DQN checkpoints. The `disrupt_0p35` candidate had a small positive paired validation-return difference against the matching control models (+0.48 points averaged across two training seeds), but its 95% paired interval included zero ([-1.03, +1.99]), and useful-revisit success fell from 0.820/0.834 for control to 0.797/0.802. Training-seed effects were inconsistent: the candidate gained +2.43 points against control seed 3090 and lost 1.47 against control seed 4090. That is not adequate evidence to replace the current selection.

The candidate checkpoints and logs are retained here for reproducibility. The existing submission checkpoints and report were not modified.

## Starting point and protocol

The control is the previously selected constraint-aware DQN, with existing model checkpoints `models/sensitivity/constrained_seed3090.pt` and `models/sensitivity/constrained_seed4090.pt`. Each was trained for 250 episodes of up to 120 steps. Its cost weights are memory 1.0, restoration 0.8, lost work 1.2, task disruption 0.45, action 0.035, invalid action 0.12, and deferred event 0.25. DQN settings were unchanged: 101-128-128-37 MLP, Adam 0.001, gamma 0.97, batch 64, replay capacity 50,000, target sync every 500 learner updates, epsilon 1.0 to 0.05 over 20,000 decisions, CPU. All DQNs used masks in exploration, greedy action selection, and replay bootstrap targets.

All candidates retained the existing environment, fixed 120-step horizon, action space, and constraint-aware transitions. Initial overflow tabs start suspended; infeasible restores/recoveries are deferred and over-budget arrivals declined. Every candidate validation rollout asserted the active memory invariant. All observed evaluation episodes had zero memory violations, and all evaluated actions were valid.

The four coefficient candidates and rationales were written to `experiment_plan.md` before training. Validation used the same 40 scenario seeds, 92000–92039, for every configuration and policy. The primary comparison across weight variants uses **canonical return**, which re-scores each candidate's exact trajectory with the original control weights. This avoids comparing raw shaped returns from different reward scales. Candidate-shaped returns are also retained. Initial screening trained each candidate from scratch with seed 3090. The two strongest screening candidates on canonical return (`lost_1p5` and `disrupt_0p35`) received a second clean training run with seed 4090.

The earlier test block 91000–91059 was examined in the preceding project experiment. It was therefore not treated as an untouched, preregistered test, and it was not used for validation selection. Since the selected model remains the original control, its already completed common-seed held-out comparison is reused here with provenance; it was not rerun.

## Configurations and validation outcomes

| Configuration | Changed weights (lost, restore, disruption) | Seed 3090 canonical return ± SD (95% CI) | Seed 4090 canonical return ± SD (95% CI) | Paired candidate-minus-control difference by training seed (95% scenario CI) |
|---|---|---:|---:|---:|
| Control | (1.20, .80, .45) | -21.53 ± 5.31 [-23.18, -19.89] | -18.21 ± 4.42 [-19.58, -16.84] | — |
| `lost_1p5` | (1.50, .80, .45) | -18.97 ± 4.38 [-20.33, -17.62] | -22.07 ± 6.27 [-24.01, -20.12] | +2.56 [+0.66, +4.46] / -3.86 [-5.67, -2.05] |
| `restore_0p6` | (1.20, .60, .45) | -20.89 ± 4.10 [-22.16, -19.62] | Not retrained | +0.64 [-1.01, +2.29] (seed 3090 only) |
| `disrupt_0p35` | (1.20, .80, .35) | -19.11 ± 5.02 [-20.66, -17.55] | -19.68 ± 5.22 [-21.30, -18.06] | +2.43 [+0.26, +4.59] / -1.47 [-3.43, +0.49] |
| `balanced` | (1.50, .70, .40) | -21.32 ± 5.15 [-22.92, -19.72] | Not retrained | +0.21 [-1.86, +2.28] (seed 3090 only) |

Intervals in the per-model columns describe episode variation on the common validation scenarios, not training-seed uncertainty. Paired intervals compare a candidate against the current model with the matching training seed on identical environment seeds. The mean of the two paired seed effects was -0.65 for `lost_1p5` and +0.48 for `disrupt_0p35`; these are descriptive averages of only two training-seed effects, not population-level confidence intervals. No candidate showed a reliable two-seed advantage. Both shortlisted candidates reversed direction across training seeds; `disrupt_0p35` was directionally positive on average but noisy and accompanied by lower useful-revisit success.

### Trade-offs

The control models had useful-revisit success 0.820 and 0.834, restoration costs 5.76 and 4.85, lost-work costs 6.42 and 5.12, disruption costs 3.74 and 2.40, and deferred costs 1.41 and 1.63.

For `disrupt_0p35`, useful-revisit success was 0.797/0.802, restoration 4.86/5.72, lost work 5.62/5.14, disruption 2.20/2.37 under the candidate's lower coefficient, and deferred costs 1.60/1.58. Under canonical scoring, the lower raw disruption amounts are reweighted to the original coefficient. The apparent mean return gain is modest and its paired uncertainty spans zero.

For `lost_1p5`, useful-revisit success was 0.805/0.810, restoration 5.06/6.44, lost-work costs 6.49/8.22 under the increased coefficient, disruption 2.88/3.92, and deferred costs 1.64/0.94. Its canonical return was better than matching control for seed 3090 (+2.56), worse for seed 4090 (-3.86), and slightly worse on pooled paired comparison.

The one-seed `restore_0p6` candidate reduced restoration cost to 4.47 and had useful-revisit success 0.825, but its canonical difference against control was uncertain. The balanced candidate did not improve canonical return in screening. Neither warranted a second training seed under the prewritten selection rule.

## Existing held-out comparison (reused)

| Policy | Return ± SD | 95% CI | Violation steps | Useful revisit | Restoration | Lost work | Disruption | Deferred |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DQN seed 3090 | -21.70 ± 4.51 | [-22.84, -20.56] | 0 | 0.824 | 5.69 | 6.73 | 3.67 | 1.41 |
| DQN seed 4090 | -16.31 ± 3.52 | [-17.20, -15.42] | 0 | 0.849 | 4.37 | 4.33 | 2.07 | 1.34 |
| LRU | -58.38 ± 4.67 | [-59.56, -57.20] | 0 | 0.886 | 16.32 | 1.69 | 36.17 | 0.00 |
| LFU | -58.43 ± 4.67 | [-59.61, -57.25] | 0 | 0.886 | 16.35 | 1.69 | 36.19 | 0.00 |
| Random | -58.55 ± 4.62 | [-59.72, -57.38] | 0 | 0.887 | 16.34 | 1.70 | 36.30 | 0.01 |

The original DQN checkpoints have higher return than the three baselines on this previously examined test block. Paired DQN-minus-baseline differences and episode-level outputs are copied from the original evaluation under `heldout_existing/`. This is the prior experiment result, not a fresh test of any reward candidate. Memory safety is enforced by the environment; it is not learned by DQN.

## Reproducibility and artifacts

From the repository root, the original virtual environment's Python can run:

```powershell
..\.venv\Scripts\python.exe -m pytest -q
..\.venv\Scripts\python.exe results\final_optimization\train_variant.py --name lost_1p5 --seed 3090 --lost 1.5
..\.venv\Scripts\python.exe results\final_optimization\evaluate_variant.py --name lost_1p5 --lost 1.5 --checkpoint 3090=results/final_optimization/checkpoints/lost_1p5_seed3090.pt --out results/final_optimization/screen_lost_1p5
..\.venv\Scripts\python.exe results\final_optimization\summarize_optimization.py
..\.venv\Scripts\python.exe results\final_optimization\plot_optimization.py
```

The remaining runs are fully specified in the per-run `training_config.json` and `evaluation_config.json` files. Candidate logs/checkpoints are in `runs/` and `checkpoints/`; per-episode and aggregate evaluations are in `screen_*`, `confirm_*`, and `control_validation/`. CSV summaries are at the experiment root. Four plots are in `plots/`.

## Limitations

There are two training seeds for the shortlisted configurations and one seed for the other two candidates. The common validation set has 40 simulated sessions; per-episode confidence intervals do not capture broad environment or training-seed variation. The baseline policies intentionally suspend aggressively and have higher task disruption, while achieving a better revisit success rate. The simulation's workload and memory units are synthetic. The prior held-out data were already examined, so claims about generalization must remain qualified. No model or report was replaced, and no Git push was performed.
