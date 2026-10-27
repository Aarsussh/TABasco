# Final optimization pass: pre-run protocol and hypotheses

## Immutable control
Use the existing constraint-aware DQN checkpoints, trained for 250 episodes with seeds 3090 and 4090, as the current control. Preserve them and all prior experiment artifacts. Environment: N=12, B=8, horizon=120, constraint_aware=True; coefficients memory=1.0, restore=.8, lost=1.2, disruption=.45, action=.035, invalid=.12, deferred=.25. DQN architecture/training settings stay at the established 101-128-128-37 MLP, Adam .001, gamma .97, batch 64, replay 50,000, target sync 500 learner updates, epsilon 1 to .05 over 20,000 decisions, 250 episodes, CPU.

## Validation protocol
Use common, fresh validation environment seeds 92000–92039 for all policies and candidates. The prior test block 91000–91059 has already been inspected in the previous experiment and is not pristine; it will only be used once at the end for the selected candidate and established control/baselines, with that limitation stated. No test results may influence candidate choice.

## Reward candidates specified before screening
The constrained active-memory invariant is retained in every run. All unchanged coefficients remain at the control values.

- `lost_1p5`: lost-work coefficient 1.50 (+25%, from 1.20). Rationale: closing a page directly charges lost unsaved-work risk and recovered pages charge residual lost-work risk; modestly increasing this term tests whether the policy retains/recoveries useful work.
- `restore_0p6`: restoration coefficient 0.60 (-25%, from .80). Rationale: explicit restore and automatic suspended revisit costs are separate from task disruption; a modest reduction tests whether the policy can keep useful pages accessible without overpaying recovery.
- `disrupt_0p35`: task-disruption coefficient .35 (-22%, from .45). Rationale: this term is action-induced interruption weighted by revisit probability/task affinity, while restoration is a separate operational cost. A modest reduction tests whether the current policy over-suspends to avoid perceived interruption.
- `balanced`: lost=1.50, restore=.70, disruption=.40. Rationale: a conservative combination of preservation emphasis and small reductions to restoration/disruption, to test interaction without removing any cost.

Screen candidates using training seed 3090 and the common validation set. If one or two appear promising on mean return with sensible revisit/cost trade-offs, retrain those with seed 4090 and compare both seeds on the same validation set. Keep all four candidates (including weak runs) in the record. No hyperparameter or architecture search is planned. New artifacts are confined to this directory.

## Selection rule
Primary criterion: common-seed validation mean return. Consider paired per-seed differences and 95% normal intervals, two training seeds' consistency, useful-revisit rate, lost-work/restoration/disruption/deferred costs, and the zero-violation invariant. Do not replace the established model unless an improvement is meaningfully supported and secondary costs are acceptable; otherwise retain it. Once chosen, run one final common-seed test comparison on 91000–91059 against existing control DQNs and the three existing baselines. The test block was seen before, so the result is a final consistency check, not a preregistered untouched estimate.
