# Data Setup

CLARK source files, timestamped articles and generated RAG responses are not
committed. The full pipeline expects them under `data/external/clark/` and
creates canonical temporal cohorts under `data/processed/`.

Expected source artifacts include:

- questions and answer-validity spans;
- official external-source mappings;
- question-to-article relevance links;
- timestamp or archive mappings for article versions.

The only committed data file is `processed/mini_temporal_retrieved.jsonl`. It is
a two-question synthetic fixture with invented entities used by the no-API smoke
test.

See [`../docs/CLARK_DATA_PIPELINE.md`](../docs/CLARK_DATA_PIPELINE.md).
