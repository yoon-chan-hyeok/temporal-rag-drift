# Results

## Frozen future transfer

The detector was calibrated once at T0 and applied to later cumulative-news
updates without retraining.

| Cohort | N | Positive | AUROC | AUPRC | Precision | Recall | F1 | Risk lift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Future question-disjoint confirmatory | 186 | 28 | **0.854** | 0.433 | 0.541 | **0.714** | **0.615** | **3.59x** |
| All future primary cases | 320 | 60 | 0.870 | 0.539 | 0.592 | 0.750 | 0.662 | 3.16x |

Confirmatory 95% question-clustered bootstrap intervals:

- AUROC: 0.769 to 0.925
- AUPRC: 0.303 to 0.614
- F1: 0.456 to 0.741
- risk lift: 2.759x to 4.843x

## Confirmatory transfer by interval

| Update end | N | Positive | AUROC | AUPRC | F1 | Risk lift |
|---|---:|---:|---:|---:|---:|---:|
| 2023-01-29 | 74 | 12 | 0.894 | 0.457 | 0.733 | 3.77x |
| 2023-07-31 | 43 | 5 | 0.792 | 0.485 | 0.500 | 3.69x |
| 2023-11-21 | 38 | 5 | 0.658 | 0.255 | 0.333 | 2.17x |
| 2024-04-19 | 31 | 6 | 0.980 | 0.915 | 0.727 | 4.13x |

The 2023-11 interval is substantially weaker. This prevents a claim that the
detector has constant performance over time.

![Frozen transfer surfaces](../assets/clark_t0_temporal_transfer_surface.png)

## P1-P5 probe

| Cohort | N | P1 accuracy | P5 accuracy | P1 failures recovered by P5 |
|---|---:|---:|---:|---:|
| New degradation | 22 | 0.219 | 1.000 | 18/18 |
| Persistent failure | 18 | 0.160 | 0.944 | 15/15 |
| Adaptive control | 22 | 0.977 | 1.000 | No P1 failure |
| Normal control | 22 | 0.972 | 1.000 | No P1 failure |

Earliest recovery candidates across failure cohorts:

| Candidate mechanism | Questions |
|---|---:|
| Retrieval coverage | 2 |
| Ranking or position sensitivity | 1 |
| Evidence extraction or context complexity | 12 |
| Evidence utilization or answer realization | 18 |

Four new-degradation and three persistent-failure questions did not fail on the
probe rerun. They are counted as no-failure-on-rerun rather than assigned a
mechanism.

![Probe accuracy](../assets/clark_probe_accuracy_by_stage.png)

## Defensible conclusion

Within this CLARK setup, a detector frozen on an early update usefully ranks
new-degradation risk on later updates. The evidence supports review
prioritization after cumulative news updates. It does not support universal
failure detection or absolute correctness certification.
