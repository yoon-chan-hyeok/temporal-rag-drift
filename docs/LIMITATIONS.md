# Limitations

## Supported

- Temporal answer distributions can be measured from fixed local CLARK
  snapshots.
- A T0-frozen two-axis detector ranked new-degradation risk on a future
  question-disjoint confirmatory cohort.
- The P1-P5 ladder can identify the first evidence intervention associated with
  recovery.

## Not supported

- A risk score proves that one answer is incorrect.
- The detector generalizes to every RAG dataset, model, prompt or retriever.
- The probe stage is always the true causal failure layer.
- Retrieval drift and generation drift are completely separated.
- A single frozen detector can be used indefinitely without monitoring its
  validity horizon.

## Statistical constraints

- The confirmatory cohort has 186 cases and 28 positive events.
- Per-update positive counts range from 5 to 12.
- Performance varies notably across the four future intervals.
- Accuracy-drop labels depend on an offline answer evaluator and CLARK answer
  validity annotations.

## Reproducibility constraints

- Original CLARK files and article text are not redistributed.
- The historical hosted-model alias may not resolve identically in the future.
- The repository includes minimum dependencies, not a byte-identical lockfile
  for every historical environment.
