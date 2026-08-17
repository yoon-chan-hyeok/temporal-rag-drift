# Curated CLARK Results

Only derived aggregate artifacts referenced by the public documentation are
included. Original questions, article text, answer samples and case-level
predictions are excluded.

## `clark_t0/`

Earlier two-axis confirmatory baseline:

- `cohort_summary.csv`: aggregate future cohort metrics and intervals
- `transition_summary.csv`: update-by-update metrics
- `report_ko.md`: archived short report

This baseline uses a narrower primary endpoint than Core4.

## `core4_ml/`

Core4 frozen-transfer experiment:

- `t0_selected_per_model.csv`: each model's T0-selected normalization and threshold
- `frozen_future_model_summary.csv`: T1-T4 aggregate performance
- `frozen_transition_summary.csv`: performance by update
- `paired_model_bootstrap.csv`: paired F1 comparisons
- `robust_z_direct_surfaces.csv`: direct-slice diagnostics

## `detector_linked_probe/`

Detector screening and P1-P5 diagnostic extension:

- `screening_performance.csv`: Additive GAM performance on all 341 future events
- `probe_group_summary.csv`: P1/P5 accuracy and recovery counts by detector outcome
- `condition_summary.csv`: aggregate accuracy/entropy/volume by probe condition
- `mechanism_counts.csv`: earliest recovery stage and candidate counts
- `report_ko.md`: detector-linked probe report and interpretation limits

P5 is an oracle condition and its recovery rate is not an operational
label-free result.
