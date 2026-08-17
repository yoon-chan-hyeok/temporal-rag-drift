# Results

## 1. Core4 frozen transfer

Each row uses the representation selected for that model from T0 only. Future
contains all 341 T1-T4 changed-question events, including persistent failures
in the negative class.

| Model | Representation | T0 F1 | Future AUROC | AUPRC | Precision | Recall | F1 | Risk lift |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| L2 logistic | robust-z | 0.691 | **0.883** | 0.617 | 0.526 | 0.833 | 0.645 | 2.991x |
| Elastic net | robust-z | 0.679 | 0.886 | 0.642 | 0.510 | 0.833 | 0.633 | 2.900x |
| Quadratic logistic | robust-z | 0.692 | 0.860 | 0.558 | 0.538 | 0.817 | **0.649** | 3.060x |
| Additive GAM | robust-z | 0.679 | 0.853 | 0.531 | 0.533 | 0.800 | 0.640 | 3.031x |
| RBF-SVM | ECDF | 0.654 | 0.865 | **0.664** | 0.506 | 0.733 | 0.599 | 2.874x |
| Extra Trees | robust-z | **0.706** | 0.867 | 0.534 | 0.528 | 0.783 | 0.631 | 3.001x |
| HistGradientBoosting | rank Gaussian | 0.667 | 0.861 | 0.506 | 0.467 | 0.817 | 0.594 | 2.652x |
| XGBoost | ECDF | 0.694 | 0.868 | 0.519 | 0.533 | 0.667 | 0.593 | 3.031x |
| MLP | robust-z | 0.640 | 0.860 | 0.563 | 0.445 | 0.883 | 0.592 | 2.531x |

Extra Trees is the formal T0 F1 winner. Quadratic logistic is the descriptive
future F1 winner. The latter cannot replace the former as a prespecified
selection because future labels were observed before making that comparison.

Paired question-cluster bootstrap differences among leading models:

| Comparison | F1 difference | 95% interval |
|---|---:|---:|
| GAM - quadratic logistic | -0.0093 | [-0.0341, 0.0090] |
| GAM - L2 logistic | -0.0057 | [-0.0409, 0.0256] |
| GAM - Extra Trees | 0.0091 | [-0.0133, 0.0349] |

All intervals include zero. The evidence supports a transferable Core4 signal,
not superiority of GAM or another single family.

## 2. Additive GAM by update

The diagnostic extension uses Additive GAM because its four-dimensional
response can be inspected through direct slices.

| Update | N | Positive | AUROC | F1 | Risk lift |
|---|---:|---:|---:|---:|---:|
| T1 | 123 | 22 | 0.847 | 0.655 | 2.95x |
| T2 | 92 | 16 | 0.861 | 0.683 | 3.22x |
| T3 | 70 | 11 | 0.851 | 0.615 | 3.39x |
| T4 | 56 | 11 | 0.821 | 0.560 | 2.55x |
| T1-T4 | 341 | 60 | **0.853** | **0.640** | **3.03x** |

Question-clustered bootstrap 95% intervals for T1-T4:

- AUROC: [0.799, 0.900]
- AUPRC: [0.408, 0.651]
- recall: [0.694, 0.897]
- F1: [0.536, 0.723]
- risk lift: [2.572x, 3.664x]

Performance weakens at T4; temporal invariance is not established.

## 3. Detector-linked probe

Screening on all 341 future events:

| TP | FP | FN | TN | AUROC | AUPRC | Precision | Recall | F1 | Risk lift |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 48 | 42 | 12 | 239 | 0.853 | 0.531 | 0.533 | 0.800 | 0.640 | 3.031x |

Probe cohort composition:

| Group | N | P1 drop reproduced | P2-P5 recovery candidate | Mean P1 accuracy | Mean P5 accuracy |
|---|---:|---:|---:|---:|---:|
| True positive | 48 | 43 | 43 | 0.254 | 0.999 |
| False negative | 12 | 12 | 12 | 0.073 | 1.000 |
| False positive | 42 | 1 | 1 | 0.676 | 0.981 |
| Matched true-negative control | 42 | 3 | 3 | 0.885 | 0.961 |

Across the 60 historical positives, 55 reproduced the drop at P1. Their
earliest recovery candidates were:

| Candidate | Events |
|---|---:|
| Retrieval coverage | 4 |
| Ranking or position | 1 |
| Evidence extraction or context complexity | 12 |
| Evidence utilization or answer realization at P5 | 37 |
| Stochastic P2 recovery without context change | 1 |
| No degradation on probe rerun | 5 |

Latest support was already present in natural top-k for 52/60 positives. This
makes pure coverage failure insufficient as the dominant explanation in this
run.

The detector alarm and a probe recovery stage were both observed for 43/60
historical positives. This 71.7% includes P5 and therefore represents an
oracle-aided diagnostic upper bound. Restricting recovery to P2-P4 gives
11/60 (18.3%), and those stages still use benchmark-selected current support.

## 4. Earlier confirmatory baseline

The repository retains a prior two-axis quadratic detector on a narrower,
question-disjoint primary subset:

| N | Positive | AUROC | AUPRC | Precision | Recall | F1 | Risk lift |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 186 | 28 | 0.854 | 0.433 | 0.541 | 0.714 | 0.615 | 3.59x |

This result and Core4 use different negative populations. They are supporting
experiments, not a before/after improvement comparison.

## Defensible conclusion

Within the measured CLARK regime, answer-distribution change contains useful
signal for prioritizing questions that newly degrade after cumulative news
updates. A T0-frozen Core4 detector retained AUROC around 0.85 and tripled the
positive rate among alarms. The probe can narrow diagnostic candidates under
controlled evidence interventions, but operational label-free root-cause
localization and absolute correctness detection remain unproven.
