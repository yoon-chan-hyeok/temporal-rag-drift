# Methods

## 1. Temporal unit

Each event contains one CLARK natural-language question, the answer valid at an
old time `x`, the answer valid at a later time `y`, and evidence retrieved from
two cumulative news snapshots:

```text
Kx = articles timestamped at or before x
Ky = articles timestamped at or before y, where x < y
```

The comparison is `stale_only(Kx)` versus `current_only(Ky)`. `Ky` retains
historical articles and adds newly available evidence; no synthetic mixed
condition is used.

## 2. Retrieval

Articles are materialized in a local SQLite FTS5 index. The same retriever is
used at both timestamps:

1. apply the snapshot time filter;
2. retrieve lexical candidates with BM25;
3. retrieve dense candidates with `BAAI/bge-large-en-v1.5`;
4. fuse rankings with reciprocal-rank fusion;
5. deduplicate by article and select top-k 10.

Top-k contexts and ranking metadata are stored before answer generation. The
audit checks timestamp leakage, duplicate articles, cardinality, current/old
answer support and support rank.

## 3. Answer generation

The fixed generation setup is:

| Setting | Value |
|---|---|
| Model | `gpt-5.6-luna` through an OpenAI-compatible API |
| Samples | 16 per question and snapshot |
| Temperature | 0.8 |
| Top-p | 0.95 |
| Max new tokens | 96 |
| Context rule | use only retrieved evidence and the stated evaluation date |
| Conflict rule | prefer the newest directly relevant passage available by that date |

The detector experiment reuses completed answer samples; fitting the ML
detector does not make additional API calls.

## 4. Semantic answer distributions

Answers are embedded with `BAAI/bge-large-en-v1.5`. Semantic equivalence
clusters use bidirectional entailment from `microsoft/deberta-large-mnli`, with
numeric mismatches forced into separate clusters.

The Core4 detector uses only temporal changes:

- **Energy distance:** geometric separation between old and current answer embeddings.
- **Cluster JS:** Jensen-Shannon divergence between old/current semantic-cluster probabilities.
- **Delta entropy:** current semantic entropy minus old semantic entropy.
- **Delta volume:** current semantic volume minus old semantic volume.

Energy and JS represent distribution shift. Delta entropy and delta volume
represent change in uncertainty. Absolute old/current entropy and volume are
excluded from Core4.

## 5. T0-fitted normalization

Each raw feature is transformed using T0 only; later updates never redefine the
reference distribution.

- ECDF: T0 empirical percentile.
- Rank Gaussian: inverse-normal transform of the T0 ECDF.
- Robust-z: `(value - T0 median) / (T0 IQR / 1.349)`, clipped to `[-8, 8]`.

Nine model families and the three representations form 27 candidates. Repeated
nested stratified folds create T0 out-of-fold predictions. Inner folds select
hyperparameters by average precision; the alarm threshold maximizes T0
out-of-fold F1. Class-balanced loss or weights handle the positive minority.

The compared families are L2 logistic, elastic net, quadratic logistic,
additive GAM, RBF-SVM, Extra Trees, histogram gradient boosting, XGBoost and
MLP. XGBoost and MLP use prespecified baseline configurations rather than the
same grid size as the other families.

## 6. Offline outcome label

Accuracy is the fraction of 16 samples matching the answer valid at each
timestamp. Labels are retrospective evaluation targets:

```text
persistent failure = accuracy_x < 0.50 and accuracy_y < 0.50
new degradation    = not persistent and accuracy_x - accuracy_y >= 0.10
target              = 1 for new degradation; 0 for every other outcome
```

Gold answers create T0 labels and evaluate later predictions. They are not
detector features. Core4 uses changed questions only; stable questions are not
used for its normalization, fitting or future score.

## 7. Frozen temporal transfer

| Split | Update | Events | Positive |
|---|---|---:|---:|
| T0 | 2021-12-22 -> 2022-08-31 | 167 | 24 |
| T1 | 2022-08-31 -> 2023-01-29 | 123 | 22 |
| T2 | 2023-01-29 -> 2023-07-31 | 92 | 16 |
| T3 | 2023-07-31 -> 2023-11-21 | 70 | 11 |
| T4 | 2023-11-21 -> 2024-04-19 | 56 | 11 |

T1-T4 contain 341 update events and 329 unique questions. T0 questions are
excluded from future questions. No future label is used to refit a
normalization, model, hyperparameter or threshold. Confidence intervals use
question-clustered bootstrap so repeated future observations of one natural
question stay together.

The earlier two-axis confirmatory baseline uses a different primary subset:
new degradation versus recovery/adaptive success, excluding persistent
failures. Its 186-case result is retained as prior evidence, not compared to
Core4 as a direct performance improvement.

## 8. Detector-linked probe

The exploratory diagnostic extension uses the frozen Core4 Additive GAM.
Screening metrics are computed on all 341 future events. The probe cohort then
contains all 60 positives, all 42 false positives and 42 risk/stratum-matched
true-negative controls.

All five conditions use the same evaluation timestamp and hide condition names
from the model:

| Stage | Intervention |
|---|---|
| P1 | Natural current top-k |
| P2 | Guarantee current support is present |
| P3 | Move current support to rank 1 |
| P4 | Supply decisive evidence only |
| P5 | Supply a compact fact card containing the current gold fact |

A historical drop is reproduced when pre-update accuracy minus P1 accuracy is
at least 0.10. Recovery requires at least 0.10 gain from P1 and a residual gap
from pre-update accuracy of at most `1/16`.

The earliest recovery stage is a failure-location candidate, not a causal
proof. P5 is an oracle upper bound because it explicitly includes the gold
current fact.
