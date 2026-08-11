"""I/O, logging, and run-directory utilities."""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

LOGGER = logging.getLogger(__name__)


def project_root() -> Path:
    """Return the repository root for this project."""
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    """Resolve a path relative to ``base_dir`` or the project root."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    root = Path(base_dir) if base_dir is not None else project_root()
    return (root / candidate).resolve()


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a ``Path``."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def write_yaml(data: Mapping[str, Any], path: str | Path) -> None:
    """Write a YAML file."""
    ensure_dir(Path(path).parent)
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(data), handle, sort_keys=False, allow_unicode=True)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dictionaries."""
    rows: list[dict[str, Any]] = []
    file_path = Path(path)
    if not file_path.exists():
        return rows
    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {file_path}:{line_number}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"JSONL row must be an object at {file_path}:{line_number}")
            rows.append(item)
    return rows


def append_jsonl(rows: Iterable[Mapping[str, Any]], path: str | Path) -> None:
    """Append dictionaries to a JSONL file."""
    ensure_dir(Path(path).parent)
    with Path(path).open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, default=str) + "\n")


def write_jsonl(rows: Iterable[Mapping[str, Any]], path: str | Path) -> None:
    """Write dictionaries to a JSONL file."""
    ensure_dir(Path(path).parent)
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, default=str) + "\n")


def write_json(data: Mapping[str, Any], path: str | Path) -> None:
    """Write an indented JSON file."""
    ensure_dir(Path(path).parent)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(dict(data), handle, ensure_ascii=False, indent=2, default=str)


def write_csv(rows: list[Mapping[str, Any]], path: str | Path) -> None:
    """Write rows to CSV, preserving the first-seen column order."""
    ensure_dir(Path(path).parent)
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def timestamped_run_dir(outputs_root: str | Path, experiment_name: str) -> Path:
    """Create a timestamped run directory."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in experiment_name)
    return ensure_dir(Path(outputs_root) / f"{stamp}_{safe_name}")


def init_run_subdirs(run_dir: str | Path) -> dict[str, Path]:
    """Create and return the standard output subdirectories."""
    root = Path(run_dir)
    subdirs = {
        "samples": root / "samples",
        "metrics": root / "metrics",
        "stats": root / "stats",
        "figures": root / "figures",
        "logs": root / "logs",
        "report": root / "report",
    }
    for directory in subdirs.values():
        ensure_dir(directory)
    return subdirs


def setup_logging(log_path: str | Path | None = None, level: int = logging.INFO) -> None:
    """Configure console and optional file logging."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_path is not None:
        ensure_dir(Path(log_path).parent)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=handlers,
        force=True,
    )


def copy_config_to_run(config: Mapping[str, Any], run_dir: str | Path) -> None:
    """Persist the effective config under the run directory."""
    write_yaml(config, Path(run_dir) / "config.yaml")


def load_run_config(run_dir: str | Path) -> dict[str, Any]:
    """Load the config snapshot from a run directory."""
    path = Path(run_dir) / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Run config not found: {path}")
    return load_yaml(path)
