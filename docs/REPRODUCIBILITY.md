# Reproducibility

## What is public

This repository includes the CLARK pipeline code, sanitized configurations,
focused tests, detector surfaces and aggregate result tables. It intentionally
excludes CLARK source records, article text, SQLite indexes, case-level
predictions, API responses, model weights and credentials.

The included aggregate tables allow result auditing. Exact scientific
reproduction requires the excluded licensed/local artifacts.

## Environment

Historical runs used Windows 11, Python 3.11 and two NVIDIA RTX 3090 GPUs.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

Install a CUDA-compatible PyTorch build before the remaining dependencies when
GPU metric computation is required. Cache:

- `BAAI/bge-large-en-v1.5`
- `microsoft/deberta-large-mnli`

Archived scientific configs use `local_files_only: true`.

## Local verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
.\.venv\Scripts\python.exe scripts\run_experiment.py --config configs\mini_temporal_mock.yaml
```

The mini experiment uses invented records, mock generation, hashing embeddings
and heuristic clustering. It is not a CLARK result.

## Stage 0: prepare private CLARK artifacts

1. Obtain CLARK through its official source and follow its data terms.
2. Store source files under `data/external/clark/`.
3. Build official question-answer-evidence linkage.
4. Materialize timestamped article text.
5. Build the local SQLite FTS5 index.
6. Build cumulative `Kx` and `Ky` top-k contexts.
7. Audit snapshot cutoffs, deduplication and support ranks.

Relevant entry points:

```powershell
.\.venv\Scripts\python.exe scripts\build_clark_official_linkage.py --help
.\.venv\Scripts\python.exe scripts\build_clark_fts_index.py --help
.\.venv\Scripts\python.exe scripts\build_clark_temporal_cohort.py --help
.\.venv\Scripts\python.exe scripts\audit_common_retrieval.py --help
```

These builders require source-specific paths and are not a raw-download
one-click command.

## Stage 1: generate temporal answer distributions

The archived launcher resumes from locally prepared CLARK cohorts. It does not
download or redistribute CLARK.

```powershell
.\run_clark_t0_temporal_transfer_luna.cmd --stage prepare
.\run_clark_t0_temporal_transfer_luna.cmd --stage cost
.\run_clark_t0_temporal_transfer_luna.cmd --stage all --confirm-api-cost
```

Hosted generation reads `OPENAI_API_KEY` from the environment. The cost stage
must be reviewed before paid sampling. Completed
`(question, condition, sample_idx)` keys are checkpointed so an interrupted run
can resume without regenerating finished answers.

## Stage 2: Core4 detector

Core4 summaries are stored under `results/core4_ml/`. Detector fitting requires
the T0 and future run directories produced by Stage 1. Normalization,
hyperparameters and threshold must be fit using T0 only; future outputs are
evaluation inputs only.

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_clark_core4_transfer.py `
  --calibration-run outputs\runs\clark_changed_primary_luna\calibration_changed `
  --future-run outputs\runs\clark_t0_temporal_transfer_luna\future_all_changed `
  --output-dir outputs\runs\clark_core4_reproduction
```

The published figures are direct robust-z slices of the frozen 4D L2,
quadratic and Additive GAM models. A surface is explanatory: each plotted axis
moves one feature pair together while holding the T0 median within-pair
contrast fixed. Actual alarms still come from all four dimensions.

## Stage 3: detector-linked P1-P5 probe

After frozen Core4 future predictions exist:

```powershell
.\run_clark_detector_linked_probe_luna.cmd --stage prepare
.\run_clark_detector_linked_probe_luna.cmd --stage cost
.\run_clark_detector_linked_probe_luna.cmd --stage all --confirm-api-cost
```

The completed run used 144 events, five conditions and 16 samples:
`144 x 5 x 16 = 11,520` generated answers. Screening is evaluated on all 341
future events; probe diagnostics are evaluated only on the selected 144-event
cohort.

## Integrity checks before publication

- run all unit tests and the synthetic smoke experiment;
- scan tracked files for `.env`, keys, tokens and passwords;
- reject absolute user paths and original CLARK text;
- verify Markdown links and referenced figures;
- compare aggregate CSVs against archived reports.
