# CLARK Data Pipeline

## Source layer

The experiment uses CLARK natural-language questions, time-valid answers and
external news evidence. CLARK data is treated as the source of question and
answer timing, while article versions are materialized locally for reproducible
retrieval.

Reference: [Language Modeling with Editable External Knowledge](https://aclanthology.org/2025.findings-naacl.168/)

## Local linkage

`build_clark_official_linkage.py` joins:

- question records;
- answer-validity spans;
- official external-source entries;
- timestamp or archive mappings;
- question-to-article relevance records.

It emits deterministic local identifiers and linkage audit summaries.

## News index

`build_clark_fts_index.py` stores timestamped article versions in SQLite FTS5.
The index records article identity, URL, title, text, publication/archive time
and chunk metadata. The index is excluded from Git because it contains
third-party content and can be large.

## Temporal retrieval

`build_clark_temporal_cohort.py` creates two cumulative retrieval views per
question:

```text
docs_x(q) = top-k from articles with timestamp <= x
docs_y(q) = top-k from articles with timestamp <= y
```

The same common retriever is used at both timestamps. It combines FTS5 BM25 and
BGE dense ranking through reciprocal-rank fusion, then applies temporal filters
and article deduplication.

## Checkpoints

The main prepared sequence is:

```text
2021-12-22 -> 2022-08-31  calibration T0
2022-08-31 -> 2023-01-29  future T1
2023-01-29 -> 2023-07-31  future T2
2023-07-31 -> 2023-11-21  future T3
2023-11-21 -> 2024-04-19  future T4
```

`prepare_clark_t0_temporal_transfer.py` excludes T0 questions from the newly
sampled confirmatory future cohort and writes hash manifests for the cohort,
config and detector.

## Retrieval audit

`audit_common_retrieval.py` checks:

- top-k cardinality;
- duplicate articles;
- old/current answer support in context;
- timestamps later than the snapshot cutoff;
- rank and support asymmetry across snapshots.

These checks are necessary because a detector result is not interpretable when
the two temporal conditions are built with inconsistent retrieval rules.
