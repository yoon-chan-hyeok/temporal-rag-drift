# Methods

## 1. Temporal unit

Each record contains one natural-language CLARK question, an old answer, a
current answer, two timestamps, and top-k evidence retrieved from cumulative
news snapshots.

```text
Kx = articles published at or before time x
Ky = articles published at or before time y, where x < y
```

The primary comparison is `stale_only` versus `current_only`. The updated corpus
therefore contains historical articles as well as newly accumulated evidence.

## 2. Retrieval

Articles are stored in a local SQLite FTS5 index. Retrieval combines:

1. BM25 lexical candidates;
2. BGE dense candidates from `BAAI/bge-large-en-v1.5`;
3. reciprocal-rank fusion;
4. temporal filtering at `time_x` or `time_y`;
5. article deduplication and top-k selection.

The main configuration uses top-k 10. Retrieved evidence and ranking metadata
are materialized before generation so that a run is reproducible.

## 3. Generation

The same prompt, generator and retrieval settings are fixed across snapshots.
Each question-condition pair is sampled 16 times with temperature 0.8 and
top-p 0.95. The prompt instructs the model to use only retrieved evidence,
respect the evaluation date, and prefer the newest relevant passage when
evidence conflicts.

## 4. Answer distributions

Answers are embedded with `BAAI/bge-large-en-v1.5`. Semantic equivalence is
estimated with bidirectional entailment from `microsoft/deberta-large-mnli`.
The resulting clusters support semantic entropy and cluster-JS computation.

Shift metrics:

- Sliced Wasserstein distance
- RBF-MMD
- Energy distance
- semantic-cluster JS divergence
- centroid gap

Uncertainty metrics:

- semantic entropy
- semantic volume
- their temporal changes from `Kx` to `Ky`

## 5. Two detector axes

Raw metrics are mapped to empirical percentiles using the T0 calibration
cohort. The actual frozen experiment uses:

```text
shift_score = mean percentile(SWD, MMD, Energy, Cluster-JS, centroid gap)
uncertainty_score = mean percentile(delta entropy, delta volume)
```

The final risk surface is a quadratic logistic regression over these two axes.

## 6. Offline outcome labels

Accuracy is the fraction of 16 answers matching the answer valid at each
timestamp. The primary event is a newly introduced degradation:

```text
accuracy_x - accuracy_y >= 0.10
```

Persistent failures are separated from the primary comparison. Gold answers
are used for calibration and retrospective evaluation, not as detector inputs
on future updates.

## 7. Frozen temporal transfer

- T0 calibration: 2021-12-22 to 2022-08-31
- detector: quadratic logistic
- frozen threshold: 0.784043
- future evaluation: four later cumulative-news updates
- future detector fitting or threshold tuning: none
- uncertainty intervals: question-clustered bootstrap, 2,000 rounds

The question-disjoint confirmatory subset excludes questions observed at T0.

## 8. Failure probe

The diagnostic ladder changes only evidence delivery:

| Stage | Intervention | Candidate interpretation if recovery starts here |
|---|---|---|
| P1 | Natural top-k | Baseline behavior |
| P2 | Current support guaranteed | Retrieval coverage |
| P3 | Current support at rank 1 | Ranking or position sensitivity |
| P4 | Decisive evidence only | Extraction or context complexity |
| P5 | Compact fact card | Evidence utilization or answer realization |

Recovery stage is an intervention-based hypothesis, not a certified causal
root cause.
