# Reproducibility

## Environment

The historical experiments were run on Windows 11, Python 3.11 and two NVIDIA
RTX 3090 GPUs.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

Install a CUDA-compatible PyTorch build before the remaining dependencies when
GPU metric computation is required.

## Models

Download and cache:

- `BAAI/bge-large-en-v1.5`
- `microsoft/deberta-large-mnli`

Archived configs use `local_files_only: true` after the first download.

## API configuration

Hosted generation reads its key from the environment:

```powershell
$env:OPENAI_API_KEY="..."
```

The launcher exposes a cost-only stage and requires `--confirm-api-cost` before
paid sampling.

## Local smoke test

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
.\.venv\Scripts\python.exe scripts\run_experiment.py `
  --config configs\mini_temporal_mock.yaml
```

This test uses no external API and does not reproduce scientific results.

## Full sequence

1. Place licensed CLARK inputs under `data/external/clark/`.
2. Build official linkage and timestamped article records.
3. Build the local SQLite FTS5 index.
4. Materialize T0 and future top-k cohorts.
5. Audit temporal cutoffs and retrieval support.
6. Freeze the T0 detector.
7. Review API cost.
8. Generate future responses and compute locked results.

```powershell
.\run_clark_t0_temporal_transfer_luna.cmd --stage prepare
.\run_clark_t0_temporal_transfer_luna.cmd --stage cost
.\run_clark_t0_temporal_transfer_luna.cmd --stage all --confirm-api-cost
```

Sampling checkpoints completed `(question, condition, sample_idx)` keys, so an
interrupted API run resumes without regenerating completed answers.

## Included and excluded artifacts

Included:

- source code and focused tests;
- synthetic smoke data;
- derived result tables;
- archived sanitized configs;
- two result figures.

Excluded:

- CLARK source files and article text;
- SQLite indexes;
- API response logs;
- model weights and local caches;
- credentials.
