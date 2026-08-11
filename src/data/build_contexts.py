"""Build retrieved contexts for freshness interventions."""

from __future__ import annotations

import re
import random
from dataclasses import dataclass
from math import gcd
from typing import Literal

from src.data.load_dataset import DatasetRecord

Condition = Literal["current_only", "stale_only", "mixed"]


@dataclass(frozen=True)
class ContextConfig:
    """Options for formatting retrieved context."""

    mixed_order: str = "concat"
    shuffle_mixed: bool = False
    include_doc_labels: bool = True
    doc_separator: str = "\n\n"
    ratio_total_docs: int = 8
    ratio_order: str = "interleave"
    allow_partial_ratio_docs: bool = True


def _label_docs(docs: list[tuple[str, str]], include_doc_labels: bool) -> list[str]:
    if not include_doc_labels:
        return [text for _, text in docs]
    counts: dict[str, int] = {}
    labeled: list[str] = []
    for source, text in docs:
        counts[source] = counts.get(source, 0) + 1
        labeled.append(f"[{source} doc {counts[source]}]\n{text}")
    return labeled


def _ratio_docs(record: DatasetRecord, condition: str, config: ContextConfig) -> list[tuple[str, str]] | None:
    """Return controlled current/stale docs for conditions like stale_25."""
    match = re.fullmatch(r"stale_(\d{1,3})(?:_(interleave|current_first|stale_first))?", condition)
    if not match:
        return None
    stale_ratio = int(match.group(1))
    ratio_order = match.group(2) or config.ratio_order
    if stale_ratio < 0 or stale_ratio > 100:
        raise ValueError(f"stale ratio must be within [0, 100]: {condition}")
    total = max(1, int(config.ratio_total_docs))
    current_count, stale_count = _ratio_counts(
        stale_ratio=stale_ratio,
        total=total,
        available_current=len(record.current_docs),
        available_stale=len(record.stale_docs),
        allow_partial=config.allow_partial_ratio_docs,
    )
    current = [("current", doc) for doc in record.current_docs[:current_count]]
    stale = [("stale", doc) for doc in record.stale_docs[:stale_count]]
    if not config.allow_partial_ratio_docs and len(current) < current_count:
        raise ValueError(f"{condition} needs {current_count} current docs, found {len(current)} for {record.id}")
    if not config.allow_partial_ratio_docs and len(stale) < stale_count:
        raise ValueError(f"{condition} needs {stale_count} stale docs, found {len(stale)} for {record.id}")
    if ratio_order == "current_first":
        return current + stale
    if ratio_order == "stale_first":
        return stale + current
    if ratio_order == "interleave":
        return _interleave_docs(current, stale)
    raise ValueError(f"Unknown ratio_order: {ratio_order}")


def _ratio_counts(
    stale_ratio: int,
    total: int,
    available_current: int,
    available_stale: int,
    allow_partial: bool,
) -> tuple[int, int]:
    """Return current/stale counts, preserving the target ratio when partially filled."""
    desired_stale = round(total * stale_ratio / 100)
    desired_current = total - desired_stale
    if not allow_partial:
        return desired_current, desired_stale
    if stale_ratio == 0:
        return min(total, available_current), 0
    if stale_ratio == 100:
        return 0, min(total, available_stale)
    divisor = gcd(desired_current, desired_stale)
    base_current = desired_current // divisor
    base_stale = desired_stale // divisor
    base_total = base_current + base_stale
    scale = min(
        total // base_total,
        available_current // base_current if base_current else total,
        available_stale // base_stale if base_stale else total,
    )
    if scale > 0:
        return base_current * scale, base_stale * scale
    return min(desired_current, available_current), min(desired_stale, available_stale)


def _interleave_docs(current: list[tuple[str, str]], stale: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Interleave ranked current/stale docs while keeping each source ranking stable."""
    if len(stale) > len(current):
        first, second = stale, current
    else:
        first, second = current, stale
    docs: list[tuple[str, str]] = []
    for index in range(max(len(first), len(second))):
        if index < len(first):
            docs.append(first[index])
        if index < len(second):
            docs.append(second[index])
    return docs


def build_context(
    record: DatasetRecord,
    condition: str,
    config: ContextConfig | None = None,
    seed: int | None = None,
) -> str:
    """Return the retrieved context for one intervention condition."""
    config = config or ContextConfig()
    condition_docs = record.metadata.get("condition_docs")
    predefined = condition_docs.get(condition) if isinstance(condition_docs, dict) else None
    if isinstance(predefined, list) and all(isinstance(item, str) for item in predefined):
        docs = [("retrieved", doc) for doc in predefined]
    else:
        ratio_docs = _ratio_docs(record, condition, config)
        if ratio_docs is not None:
            docs = ratio_docs
        elif condition == "current_only":
            docs = [("current", doc) for doc in record.current_docs]
        elif condition == "stale_only":
            docs = [("stale", doc) for doc in record.stale_docs]
        elif condition == "mixed":
            mixed_docs_raw = record.metadata.get("mixed_docs")
            if isinstance(mixed_docs_raw, list) and all(isinstance(item, str) for item in mixed_docs_raw):
                docs = [("mixed", doc) for doc in mixed_docs_raw]
            else:
                current = [("current", doc) for doc in record.current_docs]
                stale = [("stale", doc) for doc in record.stale_docs]
                if config.shuffle_mixed:
                    docs = current + stale
                    rng = random.Random(seed)
                    rng.shuffle(docs)
                elif config.mixed_order in ("concat", "current_first"):
                    docs = current + stale
                elif config.mixed_order == "stale_first":
                    docs = stale + current
                else:
                    raise ValueError(f"Unknown mixed_order: {config.mixed_order}")
        else:
            raise ValueError(f"Unknown condition: {condition}")
    return config.doc_separator.join(_label_docs(docs, config.include_doc_labels))


def parse_context_config(config: dict[str, object] | None) -> ContextConfig:
    """Parse a context config mapping into a dataclass."""
    config = config or {}
    return ContextConfig(
        mixed_order=str(config.get("mixed_order", "concat")),
        shuffle_mixed=bool(config.get("shuffle_mixed", False)),
        include_doc_labels=bool(config.get("include_doc_labels", True)),
        doc_separator=str(config.get("doc_separator", "\n\n")),
        ratio_total_docs=int(config.get("ratio_total_docs", 8)),
        ratio_order=str(config.get("ratio_order", "interleave")),
        allow_partial_ratio_docs=bool(config.get("allow_partial_ratio_docs", True)),
    )
