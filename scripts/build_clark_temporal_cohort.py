"""Build CLARK/Unified Clark timepoint cohorts for cumulative-news RAG experiments.

This script converts CLARK-style timestamped questions plus text-bearing article
archives into the canonical JSONL format already used by this repository:

- ``gold_answer`` is the answer valid at ``time_y`` (the newer timepoint)
- ``stale_answer`` is the answer valid at ``time_x`` (the older timepoint)
- ``current_docs`` are top-k retrieved passages from articles with timestamp ``<= time_y``
- ``stale_docs`` are top-k retrieved passages from articles with timestamp ``<= time_x``

CLARK is naturally cumulative: the newer context
already contains older news. For that reason, the intended run configuration is
usually ``current_only`` vs ``stale_only`` without a separate ``mixed`` arm.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib.parse import urlsplit, urlunsplit

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.io import read_jsonl, write_jsonl
from src.utils.text import compact_whitespace, normalize_text, stable_text_hash
from src.retrieval.common_hybrid import (
    DenseEmbeddingCache,
    HybridRetrievalConfig,
    SQLiteHybridRetriever,
    format_retrieved_doc,
)


QUESTION_COL = "Question"
ANSWER_COL = "Answer"
QUESTION_TYPE_COL = "Question type"
START_COL = "Start timestamp"
END_COL = "End timestamp"
KNOWN_START_COL = "Known start timestamp"
KNOWN_END_COL = "Known end timestamp"
ANSWER_CHOICES_COL = "Answer choices"

TIMESTAMP_PATTERNS = (
    "timestamp",
    "article_timestamp",
    "published",
    "published_at",
    "publish_date",
    "date",
    "datetime",
    "retrieved_at",
    "archive_timestamp",
)
TEXT_PATTERNS = (
    "text",
    "article_text",
    "body",
    "content",
    "page_content",
    "source_text",
    "main_text",
)
TITLE_PATTERNS = (
    "title",
    "headline",
    "article_title",
    "source_title",
    "page_title",
)
URL_PATTERNS = (
    "archive_url",
    "url",
    "source_url",
    "link",
    "href",
)
SOURCE_HINT_PATTERNS = (
    "question",
    "questions",
    "query",
    "subject",
    "entity",
)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "hold",
    "holds",
    "how",
    "in",
    "is",
    "its",
    "member",
    "of",
    "on",
    "or",
    "position",
    "that",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
}


@dataclass(frozen=True)
class QuestionSpan:
    row_index: int
    question: str
    answer: str
    question_type: str
    answer_choices: str
    start_ts: pd.Timestamp | None
    end_ts: pd.Timestamp | None
    known_start_ts: pd.Timestamp | None
    known_end_ts: pd.Timestamp | None

    def active_range(self, mode: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        if mode == "known":
            start = self.known_start_ts or self.start_ts
            end = self.known_end_ts or self.end_ts
            return start, end
        return self.start_ts, self.end_ts


@dataclass(frozen=True)
class EvidenceDoc:
    timestamp: pd.Timestamp
    title: str
    text: str
    url: str
    source_hint: str
    raw: dict[str, Any]


@dataclass(frozen=True)
class Passage:
    article_id: str
    article_timestamp: pd.Timestamp
    title: str
    url: str
    source_url: str
    text: str
    chunk_index: int
    token_counts: Counter[str]
    title_tokens: set[str]
    doc_len: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CLARK/Unified Clark timepoint cohorts.")
    parser.add_argument("--questions-csv", required=True, help="Path to CLARK questions.csv.")
    parser.add_argument(
        "--evidence-file",
        required=True,
        help="Path to a text-bearing CLARK evidence file, ideally external_sources.json.",
    )
    parser.add_argument(
        "--timestamp-map",
        default=None,
        help="Optional timestamp_to_questions.json path used to restrict to official checkpoint questions.",
    )
    parser.add_argument(
        "--url-timestamp-map",
        default=None,
        help="Optional url_timestamp_to_archive_url.jsonl file for archive URL normalization.",
    )
    parser.add_argument("--time-x", required=True, help="Older evaluation timestamp (ISO-8601).")
    parser.add_argument("--time-y", required=True, help="Newer evaluation timestamp (ISO-8601).")
    parser.add_argument("--output", required=True, help="Output canonical JSONL path.")
    parser.add_argument("--output-changed", default=None, help="Optional changed-only JSONL path.")
    parser.add_argument("--output-stable", default=None, help="Optional stable-only JSONL path.")
    parser.add_argument("--time-mode", choices=("known", "true"), default="known")
    parser.add_argument("--selection", choices=("intersection", "union"), default="intersection")
    parser.add_argument("--change-label", choices=("all", "changed", "stable"), default="all")
    parser.add_argument("--top-k", type=int, default=10, help="Retrieved passages per timepoint.")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--question-type", action="append", default=None, help="Optional question_type filter.")
    parser.add_argument("--chunk-chars", type=int, default=1200)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument("--max-doc-chars", type=int, default=2400)
    parser.add_argument("--min-score", type=float, default=0.0, help="Minimum lexical score to keep a passage.")
    parser.add_argument(
        "--retrieval-method",
        choices=("lexical", "bm25", "hybrid", "common_hybrid"),
        default="lexical",
        help=(
            "Passage retrieval scorer. 'hybrid' is the legacy lexical+BM25 blend; "
            "'common_hybrid' uses shared BM25 candidates+dense+RRF."
        ),
    )
    parser.add_argument("--bm25-k1", type=float, default=1.2, help="BM25 k1 parameter.")
    parser.add_argument("--bm25-b", type=float, default=0.75, help="BM25 b parameter.")
    parser.add_argument(
        "--hybrid-alpha",
        type=float,
        default=0.5,
        help="Weight on lexical score when retrieval-method=hybrid (0..1).",
    )
    parser.add_argument(
        "--official-linkage-file",
        default=None,
        help="Official question_article_qrels.jsonl for provenance and support-hit auditing.",
    )
    parser.add_argument(
        "--common-index",
        default=None,
        help="CLARK SQLite FTS5 index required by retrieval-method=common_hybrid.",
    )
    parser.add_argument("--dense-model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--dense-device", default=None)
    parser.add_argument("--dense-batch-size", type=int, default=32)
    parser.add_argument("--dense-local-files-only", action="store_true")
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--sparse-weight", type=float, default=1.0)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument(
        "--article-dedup",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--common-use-title-hint",
        action="store_true",
        help="Use a target-title hint in common retrieval. Off by default to avoid asymmetric leakage.",
    )
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (ROOT / candidate).resolve()


def parse_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        timestamp = pd.to_datetime(text, utc=True, errors="coerce")
    except Exception:
        return None
    if pd.isna(timestamp):
        return None
    return timestamp


def infer_timestamp_from_url(url: str) -> pd.Timestamp | None:
    if not url:
        return None
    wayback = re.search(r"/web/(\d{14})/", url)
    if wayback:
        return parse_timestamp(wayback.group(1))
    for pattern in (
        r"/(20\d{2})/(0[1-9]|1[0-2])/(0[1-9]|[12]\d|3[01])/",
        r"(20\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])",
        r"(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])",
    ):
        match = re.search(pattern, url)
        if not match:
            continue
        groups = match.groups()
        if len(groups) >= 3:
            return parse_timestamp(f"{groups[0]}-{groups[1]}-{groups[2]}")
    return None


def infer_timestamp_from_text(text: str) -> pd.Timestamp | None:
    if not text:
        return None
    head = "\n".join(text.splitlines()[:25])[:2500]
    patterns = [
        r"\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}\b",
        r"\b[A-Za-z]{3,9}\s+\d{1,2},\s+\d{4}\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{4}/\d{2}/\d{2}\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, head):
            timestamp = parse_timestamp(match.group(0))
            if timestamp is not None:
                return timestamp
    return None


def load_question_spans(path: Path) -> dict[str, list[QuestionSpan]]:
    frame = pd.read_csv(path)
    spans: dict[str, list[QuestionSpan]] = {}
    for row_index, row in frame.iterrows():
        question = clean_text(row.get(QUESTION_COL))
        answer = clean_text(row.get(ANSWER_COL))
        if not question or not answer:
            continue
        spans.setdefault(question, []).append(
            QuestionSpan(
                row_index=int(row_index),
                question=question,
                answer=answer,
                question_type=clean_text(row.get(QUESTION_TYPE_COL)),
                answer_choices=clean_text(row.get(ANSWER_CHOICES_COL)),
                start_ts=parse_timestamp(row.get(START_COL)),
                end_ts=parse_timestamp(row.get(END_COL)),
                known_start_ts=parse_timestamp(row.get(KNOWN_START_COL)),
                known_end_ts=parse_timestamp(row.get(KNOWN_END_COL)),
            )
        )
    for question_spans in spans.values():
        question_spans.sort(
            key=lambda span: (
                span.known_start_ts or span.start_ts or pd.Timestamp.min.tz_localize("UTC"),
                span.answer,
            )
        )
    return spans


def load_timestamp_map(path: Path) -> dict[str, list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    mapping: dict[str, list[str]] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, list):
                mapping[str(key)] = [str(item).strip() for item in value if str(item).strip()]
    return mapping


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = compact_whitespace(str(value))
    return "" if text.lower() == "nan" else text


def infer_title_from_text(text: str, max_chars: int = 180) -> str:
    if not text:
        return ""
    for line in text.splitlines():
        line = compact_whitespace(line)
        if len(line) >= 8:
            return line[:max_chars]
    return compact_whitespace(text)[:max_chars]


def nested_items(value: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            new_prefix = f"{prefix}.{key_text}" if prefix else key_text
            yield new_prefix, child
            yield from nested_items(child, new_prefix)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            new_prefix = f"{prefix}[{index}]"
            yield new_prefix, child
            yield from nested_items(child, new_prefix)


def pick_first_by_path(record: dict[str, Any], patterns: Iterable[str]) -> str:
    lowered = tuple(pattern.lower() for pattern in patterns)
    for path, value in nested_items(record):
        if not isinstance(value, (str, int, float)):
            continue
        path_lower = path.lower()
        if any(pattern in path_lower for pattern in lowered):
            text = clean_text(value)
            if text:
                return text
    return ""


def pick_longest_text(record: dict[str, Any], min_chars: int = 80) -> str:
    best = ""
    for _, value in nested_items(record):
        if isinstance(value, str):
            text = compact_whitespace(value)
            if len(text) >= min_chars and len(text) > len(best):
                best = text
    return best


def iter_json_items(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        for row in read_jsonl(path):
            yield row
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item
    elif isinstance(data, dict):
        if all(isinstance(value, dict) for value in data.values()):
            for value in data.values():
                yield value
        else:
            yield data


def iter_csv_items(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield dict(row)


def load_url_timestamp_archive_map(path: Path | None) -> dict[tuple[str, str], str]:
    if path is None or not path.exists():
        return {}
    mapping: dict[tuple[str, str], str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            url = clean_text(row.get("url"))
            timestamp = clean_text(row.get("timestamp") or row.get("source_timestamp"))
            archive_url = clean_text(row.get("archive_url"))
            if url and timestamp and archive_url:
                mapping[(url, timestamp)] = archive_url
    return mapping


def load_external_sources_json(
    path: Path,
    max_doc_chars: int,
    archive_map: dict[tuple[str, str], str],
) -> list[EvidenceDoc]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("external_sources.json must be a JSON object keyed by URL.")
    docs: list[EvidenceDoc] = []
    seen: set[tuple[str, str, str]] = set()
    for url, versions in data.items():
        if not isinstance(versions, dict):
            continue
        url_text = clean_text(url)
        for timestamp_key, payload in versions.items():
            if not isinstance(payload, dict):
                continue
            timestamp_text = clean_text(payload.get("source_timestamp")) or clean_text(timestamp_key)
            timestamp = parse_timestamp(timestamp_text)
            if timestamp is None:
                continue
            text = compact_whitespace(clean_text(payload.get("source_text")))
            if max_doc_chars > 0 and len(text) > max_doc_chars:
                text = text[: max_doc_chars].rsplit(" ", 1)[0].strip()
            if len(text) < 80:
                continue
            archive_url = (
                clean_text(payload.get("archive_url"))
                or archive_map.get((url_text, timestamp_text), "")
                or url_text
            )
            title = infer_title_from_text(text)
            signature = (timestamp.isoformat(), archive_url or url_text, stable_text_hash(text[:400]))
            if signature in seen:
                continue
            seen.add(signature)
            docs.append(
                EvidenceDoc(
                    timestamp=timestamp,
                    title=title,
                    text=text,
                    url=archive_url or url_text,
                    source_hint=url_text,
                    raw={
                        "url": url_text,
                        "archive_url": archive_url,
                        "source_timestamp": timestamp_text,
                    },
                )
            )
    docs.sort(key=lambda doc: (doc.timestamp, doc.title, doc.url))
    return docs


def load_evidence_docs(
    path: Path,
    max_doc_chars: int,
    archive_map: dict[tuple[str, str], str] | None = None,
) -> list[EvidenceDoc]:
    archive_map = archive_map or {}
    if path.name == "external_sources.json":
        return load_external_sources_json(path, max_doc_chars=max_doc_chars, archive_map=archive_map)
    if path.suffix.lower() in {".json", ".jsonl"}:
        items = iter_json_items(path)
    elif path.suffix.lower() == ".csv":
        items = iter_csv_items(path)
    else:
        raise ValueError(f"Unsupported evidence file extension: {path.suffix}")

    docs: list[EvidenceDoc] = []
    seen: set[tuple[str, str, str]] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        title = pick_first_by_path(item, TITLE_PATTERNS)
        text = pick_first_by_path(item, TEXT_PATTERNS) or pick_longest_text(item)
        text = compact_whitespace(text)
        if max_doc_chars > 0 and len(text) > max_doc_chars:
            text = text[: max_doc_chars].rsplit(" ", 1)[0].strip()
        if len(text) < 80:
            continue
        url = pick_first_by_path(item, URL_PATTERNS)
        timestamp = (
            parse_timestamp(pick_first_by_path(item, TIMESTAMP_PATTERNS))
            or infer_timestamp_from_url(url)
            or infer_timestamp_from_text(text)
        )
        if timestamp is None:
            continue
        source_hint = pick_first_by_path(item, SOURCE_HINT_PATTERNS)
        signature = (timestamp.isoformat(), title, stable_text_hash(text[:400]))
        if signature in seen:
            continue
        seen.add(signature)
        docs.append(
            EvidenceDoc(
                timestamp=timestamp,
                title=title,
                text=text,
                url=url,
                source_hint=source_hint,
                raw=item,
            )
        )
    docs.sort(key=lambda doc: (doc.timestamp, doc.title, doc.url))
    return docs


def chunk_text(text: str, chunk_chars: int, overlap: int) -> list[str]:
    text = compact_whitespace(text)
    if not text:
        return []
    if len(text) <= chunk_chars:
        return [text]
    if chunk_chars <= overlap:
        overlap = max(0, chunk_chars // 5)
    chunks: list[str] = []
    start = 0
    step = max(1, chunk_chars - overlap)
    while start < len(text):
        raw = text[start : start + chunk_chars]
        if start + chunk_chars < len(text):
            raw = raw.rsplit(" ", 1)[0]
        chunk = raw.strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_chars >= len(text):
            break
        start += step
    return chunks


def build_passages(docs: list[EvidenceDoc], chunk_chars: int, overlap: int) -> list[Passage]:
    passages: list[Passage] = []
    for doc_index, doc in enumerate(docs):
        article_id = stable_text_hash(f"{doc_index}:{doc.url}:{doc.title}:{doc.timestamp.isoformat()}")
        title_tokens = set(tokenize_for_search(doc.title))
        for chunk_index, chunk in enumerate(chunk_text(doc.text, chunk_chars=chunk_chars, overlap=overlap), start=1):
            token_counts = Counter(tokenize_for_search(chunk))
            if not token_counts:
                continue
            passages.append(
                Passage(
                    article_id=article_id,
                    article_timestamp=doc.timestamp,
                    title=doc.title,
                    url=doc.url,
                    source_url=doc.source_hint or doc.url,
                    text=chunk,
                    chunk_index=chunk_index,
                    token_counts=token_counts,
                    title_tokens=title_tokens,
                    doc_len=sum(token_counts.values()),
                )
            )
    return passages


def tokenize_for_search(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9_'-]*", text.lower())
    cleaned: list[str] = []
    for token in tokens:
        token = token.strip("'-_")
        token = normalize_search_token(token)
        if len(token) < 2 or token in STOPWORDS:
            continue
        cleaned.append(token)
    return cleaned


def normalize_search_token(token: str) -> str:
    if token.endswith("ing") and len(token) > 5:
        token = token[:-3]
    elif token.endswith("ed") and len(token) > 4:
        token = token[:-2]
    if token.endswith("ies") and len(token) > 5:
        token = token[:-3] + "y"
    elif token.endswith("s") and len(token) > 4:
        token = token[:-1]
    return token


def build_idf(passages: list[Passage]) -> dict[str, float]:
    df_counts: Counter[str] = Counter()
    for passage in passages:
        df_counts.update(set(passage.token_counts))
    total = max(1, len(passages))
    return {
        token: math.log((total + 1) / (count + 1)) + 1.0
        for token, count in df_counts.items()
    }


def build_bm25_stats(passages: list[Passage]) -> tuple[dict[str, float], float]:
    df_counts: Counter[str] = Counter()
    total_doc_len = 0
    for passage in passages:
        df_counts.update(set(passage.token_counts))
        total_doc_len += passage.doc_len
    total = max(1, len(passages))
    avg_doc_len = total_doc_len / total
    idf = {
        token: math.log(1.0 + ((total - count + 0.5) / (count + 0.5)))
        for token, count in df_counts.items()
    }
    return idf, avg_doc_len


def lexical_score(question: str, passage: Passage, idf: dict[str, float]) -> float:
    question_tokens = tokenize_for_search(question)
    if not question_tokens:
        return 0.0
    score = 0.0
    for token in question_tokens:
        weight = float(idf.get(token, 1.0))
        tf = passage.token_counts.get(token, 0)
        if tf:
            score += weight * min(tf, 3)
        if token in passage.title_tokens:
            score += 0.35 * weight
    return score / math.sqrt(max(sum(passage.token_counts.values()), 1))


def bm25_score(
    question: str,
    passage: Passage,
    idf: dict[str, float],
    avg_doc_len: float,
    k1: float,
    b: float,
) -> float:
    question_tokens = tokenize_for_search(question)
    if not question_tokens:
        return 0.0
    score = 0.0
    doc_len = max(passage.doc_len, 1)
    norm = k1 * (1.0 - b + b * (doc_len / max(avg_doc_len, 1e-6)))
    for token in question_tokens:
        tf = passage.token_counts.get(token, 0)
        if not tf:
            continue
        token_idf = float(idf.get(token, 0.0))
        score += token_idf * (tf * (k1 + 1.0)) / (tf + norm)
        if token in passage.title_tokens:
            score += 0.35 * max(token_idf, 0.1)
    return score


def minmax_normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if math.isclose(lo, hi):
        if math.isclose(hi, 0.0):
            return [0.0 for _ in values]
        return [1.0 for _ in values]
    scale = hi - lo
    return [(value - lo) / scale for value in values]


def active_answer(spans: list[QuestionSpan], when: pd.Timestamp, mode: str) -> QuestionSpan | None:
    chosen: QuestionSpan | None = None
    chosen_start: pd.Timestamp | None = None
    for span in spans:
        start_ts, end_ts = span.active_range(mode)
        if start_ts is None:
            continue
        if when < start_ts:
            continue
        if end_ts is not None and when >= end_ts:
            continue
        if chosen is None or (chosen_start is None or start_ts >= chosen_start):
            chosen = span
            chosen_start = start_ts
    return chosen


def format_passage(passage: Passage, rank: int, score: float) -> str:
    return "\n".join(
        [
            f"[retrieved rank {rank}]",
            f"Timestamp: {passage.article_timestamp.isoformat()}",
            f"Title: {passage.title}",
            f"URL: {passage.url}",
            f"Source URL: {passage.source_url}",
            f"Article ID: {passage.article_id}",
            f"Chunk: {passage.chunk_index}",
            f"Retrieval score: {score:.4f}",
            passage.text,
        ]
    )


def retrieve_top_k(
    question: str,
    passages: list[Passage],
    lexical_idf: dict[str, float],
    bm25_idf: dict[str, float],
    avg_doc_len: float,
    cutoff: pd.Timestamp,
    top_k: int,
    min_score: float,
    retrieval_method: str,
    bm25_k1: float,
    bm25_b: float,
    hybrid_alpha: float,
) -> tuple[list[str], list[dict[str, Any]]]:
    candidates: list[Passage] = []
    for passage in passages:
        if passage.article_timestamp > cutoff:
            continue
        candidates.append(passage)
    if not candidates:
        return [], []

    lexical_scores = [lexical_score(question, passage, lexical_idf) for passage in candidates]
    bm25_scores = [
        bm25_score(
            question,
            passage,
            bm25_idf,
            avg_doc_len=avg_doc_len,
            k1=bm25_k1,
            b=bm25_b,
        )
        for passage in candidates
    ]
    if retrieval_method == "lexical":
        final_scores = lexical_scores
    elif retrieval_method == "bm25":
        final_scores = bm25_scores
    else:
        alpha = min(max(hybrid_alpha, 0.0), 1.0)
        lexical_norm = minmax_normalize(lexical_scores)
        bm25_norm = minmax_normalize(bm25_scores)
        final_scores = [
            alpha * lexical_value + (1.0 - alpha) * bm25_value
            for lexical_value, bm25_value in zip(lexical_norm, bm25_norm)
        ]

    scored: list[tuple[float, Passage, float, float]] = []
    for passage, lexical_value, bm25_value, final_value in zip(candidates, lexical_scores, bm25_scores, final_scores):
        if final_value >= min_score:
            scored.append((final_value, passage, lexical_value, bm25_value))
    scored.sort(key=lambda item: (item[0], item[1].article_timestamp.value), reverse=True)
    top = scored[:top_k]
    formatted = [
        format_passage(passage, rank=index, score=score)
        for index, (score, passage, _, _) in enumerate(top, start=1)
    ]
    metadata_rows = [
        {
            "rank": index,
            "score": float(score),
            "lexical_score": float(lexical_value),
            "bm25_score": float(bm25_value),
            "retrieval_method": retrieval_method,
            "timestamp": passage.article_timestamp.isoformat(),
            "title": passage.title,
            "url": passage.url,
            "source_url": passage.source_url,
            "article_id": passage.article_id,
            "chunk_index": passage.chunk_index,
        }
        for index, (score, passage, lexical_value, bm25_value) in enumerate(top, start=1)
    ]
    return formatted, metadata_rows


def load_official_linkage(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    qrels: dict[int, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if "question_row_index" not in row:
            continue
        qrels[int(row["question_row_index"])] = row
    return qrels


def normalize_url_for_match(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    parsed = urlsplit(text)
    hostname = (parsed.hostname or "").lower()
    if hostname.startswith("www."):
        hostname = hostname[4:]
    netloc = hostname
    if parsed.port:
        netloc = f"{hostname}:{parsed.port}"
    path = parsed.path.rstrip("/") or "/"
    return urlunsplit((parsed.scheme.lower(), netloc, path, parsed.query, ""))


def official_support_hits(
    qrel: dict[str, Any] | None,
    retrieval_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    support_urls = sorted(
        {
            normalize_url_for_match(url)
            for url in (qrel or {}).get("support_urls", [])
            if normalize_url_for_match(url)
        }
    )
    support_set = set(support_urls)
    matched: list[dict[str, Any]] = []
    for row in retrieval_rows:
        row_urls = {
            normalize_url_for_match(str(row.get("source_url") or "")),
            normalize_url_for_match(str(row.get("url") or "")),
        }
        overlap = sorted((row_urls - {""}) & support_set)
        if overlap:
            matched.append(
                {
                    "rank": int(row.get("rank", len(matched) + 1)),
                    "matched_urls": overlap,
                    "document_id": str(row.get("document_id") or row.get("article_id") or ""),
                    "chunk_index": row.get("chunk_index"),
                }
            )
    return {
        "mapping_status": (qrel or {}).get("mapping_status", "missing_qrel"),
        "support_urls": support_urls,
        "support_url_count": len(support_urls),
        "hit_at_k": bool(matched),
        "hit_count": len(matched),
        "best_rank": min((row["rank"] for row in matched), default=None),
        "matches": matched,
    }


def normalized_equal(left: str, right: str) -> bool:
    return normalize_text(left) == normalize_text(right)


def allowed_questions_for_checkpoint(
    timestamp_map: dict[str, list[str]] | None,
    time_x_text: str,
    time_y_text: str,
    selection: str,
) -> set[str] | None:
    if not timestamp_map:
        return None
    qs_x = set(timestamp_map.get(time_x_text, []))
    qs_y = set(timestamp_map.get(time_y_text, []))
    if selection == "union":
        return qs_x | qs_y
    return qs_x & qs_y


def build_record(
    question: str,
    x_span: QuestionSpan,
    y_span: QuestionSpan,
    time_x: pd.Timestamp,
    time_y: pd.Timestamp,
    stale_docs: list[str],
    current_docs: list[str],
    stale_meta: list[dict[str, Any]],
    current_meta: list[dict[str, Any]],
    time_mode: str,
    x_qrel: dict[str, Any] | None = None,
    y_qrel: dict[str, Any] | None = None,
    retrieval_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    change_label = "changed" if not normalized_equal(x_span.answer, y_span.answer) else "stable"
    question_id = (
        f"clark_{time_x.strftime('%Y%m%d')}_{time_y.strftime('%Y%m%d')}_"
        f"{stable_text_hash(question)}"
    )
    metadata = {
        "source": "clark_news",
        "question_type": y_span.question_type or x_span.question_type,
        "answer_choices_y": y_span.answer_choices,
        "answer_choices_x": x_span.answer_choices,
        "time_x": time_x.isoformat(),
        "time_y": time_y.isoformat(),
        "time_mode": time_mode,
        "change_label": change_label,
        "probe_status": f"clark_{change_label}_timepoint_pair",
        "stale_answer": x_span.answer,
        "old_answer": x_span.answer,
        "current_answer": y_span.answer,
        "old_answers": [x_span.answer],
        "current_answers": [y_span.answer],
        "x_question_type": x_span.question_type,
        "y_question_type": y_span.question_type,
        "retrieval": {
            "stale": stale_meta,
            "current": current_meta,
        },
        "retrieval_method": stale_meta[0].get("retrieval_method") if stale_meta else "",
        "retrieval_config": retrieval_config or {},
        "retrieval_counts": {
            "stale_docs": len(stale_docs),
            "current_docs": len(current_docs),
        },
        "official_linkage": {
            "stale": x_qrel,
            "current": y_qrel,
        },
        "official_support_hits": {
            "stale": official_support_hits(x_qrel, stale_meta),
            "current": official_support_hits(y_qrel, current_meta),
        },
    }
    return {
        "id": question_id,
        "question": question,
        "gold_answer": y_span.answer,
        "stale_answer": x_span.answer,
        "gold_old": [x_span.answer],
        "gold_new": [y_span.answer],
        "current_docs": current_docs,
        "stale_docs": stale_docs,
        "metadata": metadata,
    }


def main() -> None:
    args = parse_args()
    questions_csv = resolve(args.questions_csv)
    evidence_file = resolve(args.evidence_file)
    timestamp_map_path = resolve(args.timestamp_map) if args.timestamp_map else None
    url_timestamp_map_path = resolve(args.url_timestamp_map) if args.url_timestamp_map else None
    output = resolve(args.output)
    output_changed = resolve(args.output_changed) if args.output_changed else None
    output_stable = resolve(args.output_stable) if args.output_stable else None
    official_linkage_path = (
        resolve(args.official_linkage_file) if args.official_linkage_file else None
    )
    common_index_path = resolve(args.common_index) if args.common_index else None

    time_x = parse_timestamp(args.time_x)
    time_y = parse_timestamp(args.time_y)
    if time_x is None or time_y is None:
        raise ValueError("Both --time-x and --time-y must be valid timestamps.")
    if time_x >= time_y:
        raise ValueError("--time-x must be earlier than --time-y.")

    histories = load_question_spans(questions_csv)
    timestamp_map = load_timestamp_map(timestamp_map_path) if timestamp_map_path else None
    allowed = allowed_questions_for_checkpoint(timestamp_map, args.time_x, args.time_y, args.selection)
    official_linkage = load_official_linkage(official_linkage_path)

    evidence_docs: list[EvidenceDoc] = []
    passages: list[Passage] = []
    lexical_idf: dict[str, float] = {}
    bm25_idf: dict[str, float] = {}
    avg_doc_len = 0.0
    common_retriever: SQLiteHybridRetriever | None = None
    retrieval_config: dict[str, Any] = {
        "method": args.retrieval_method,
        "top_k": int(args.top_k),
    }
    if args.retrieval_method == "common_hybrid":
        if common_index_path is None or not common_index_path.exists():
            raise FileNotFoundError(
                "retrieval-method=common_hybrid requires an existing --common-index."
            )
        dense_cache = DenseEmbeddingCache(
            args.dense_model,
            batch_size=int(args.dense_batch_size),
            device=args.dense_device,
            local_files_only=bool(args.dense_local_files_only),
        )
        common_retriever = SQLiteHybridRetriever(
            common_index_path,
            dense_cache,
            HybridRetrievalConfig(
                candidate_k=int(args.candidate_k),
                top_k=int(args.top_k),
                rrf_k=int(args.rrf_k),
                sparse_weight=float(args.sparse_weight),
                dense_weight=float(args.dense_weight),
                article_dedup=bool(args.article_dedup),
            ),
        )
        retrieval_config.update(
            {
                "index": str(common_index_path),
                "candidate_k": int(args.candidate_k),
                "rrf_k": int(args.rrf_k),
                "sparse_weight": float(args.sparse_weight),
                "dense_weight": float(args.dense_weight),
                "article_dedup": bool(args.article_dedup),
                "dense_model": args.dense_model,
                "dense_device": args.dense_device,
                "dense_local_files_only": bool(args.dense_local_files_only),
                "title_hint": bool(args.common_use_title_hint),
            }
        )
    else:
        archive_map = load_url_timestamp_archive_map(url_timestamp_map_path)
        evidence_docs = load_evidence_docs(
            evidence_file,
            max_doc_chars=int(args.max_doc_chars),
            archive_map=archive_map,
        )
        if not evidence_docs:
            raise ValueError(
                "No text-bearing evidence documents were found. "
                "For CLARK, prefer external_sources.json together with url_timestamp_to_archive_url.jsonl."
            )
        passages = build_passages(
            evidence_docs,
            chunk_chars=int(args.chunk_chars),
            overlap=int(args.chunk_overlap),
        )
        if not passages:
            raise ValueError("No retrievable passages were built from the evidence file.")
        lexical_idf = build_idf(passages)
        bm25_idf, avg_doc_len = build_bm25_stats(passages)

    type_filter = {value.lower() for value in (args.question_type or [])}
    changed_records: list[dict[str, Any]] = []
    stable_records: list[dict[str, Any]] = []
    skipped_missing_answers = 0
    skipped_filter = 0

    ordered_questions = sorted(histories.keys())
    for question in ordered_questions:
        if allowed is not None and question not in allowed:
            continue
        spans = histories[question]
        x_span = active_answer(spans, time_x, mode=args.time_mode)
        y_span = active_answer(spans, time_y, mode=args.time_mode)
        if x_span is None or y_span is None:
            skipped_missing_answers += 1
            continue
        question_type = (y_span.question_type or x_span.question_type).lower()
        if type_filter and question_type not in type_filter:
            skipped_filter += 1
            continue
        question_change_label = (
            "changed"
            if not normalized_equal(x_span.answer, y_span.answer)
            else "stable"
        )
        if args.change_label != "all" and question_change_label != args.change_label:
            continue
        if (
            args.max_questions is not None
            and len(changed_records) + len(stable_records) >= int(args.max_questions)
        ):
            break

        if common_retriever is not None:
            title_hint = "" if not args.common_use_title_hint else question
            stale_meta = common_retriever.retrieve(
                question,
                cutoff=time_x.isoformat(),
                title=title_hint,
            )
            current_meta = common_retriever.retrieve(
                question,
                cutoff=time_y.isoformat(),
                title=title_hint,
            )
            stale_docs = [
                format_retrieved_doc(row, rank=index)
                for index, row in enumerate(stale_meta, start=1)
            ]
            current_docs = [
                format_retrieved_doc(row, rank=index)
                for index, row in enumerate(current_meta, start=1)
            ]
        else:
            stale_docs, stale_meta = retrieve_top_k(
                question,
                passages,
                lexical_idf,
                bm25_idf,
                avg_doc_len,
                cutoff=time_x,
                top_k=int(args.top_k),
                min_score=float(args.min_score),
                retrieval_method=args.retrieval_method,
                bm25_k1=float(args.bm25_k1),
                bm25_b=float(args.bm25_b),
                hybrid_alpha=float(args.hybrid_alpha),
            )
            current_docs, current_meta = retrieve_top_k(
                question,
                passages,
                lexical_idf,
                bm25_idf,
                avg_doc_len,
                cutoff=time_y,
                top_k=int(args.top_k),
                min_score=float(args.min_score),
                retrieval_method=args.retrieval_method,
                bm25_k1=float(args.bm25_k1),
                bm25_b=float(args.bm25_b),
                hybrid_alpha=float(args.hybrid_alpha),
            )
        if not stale_docs or not current_docs:
            continue

        record = build_record(
            question=question,
            x_span=x_span,
            y_span=y_span,
            time_x=time_x,
            time_y=time_y,
            stale_docs=stale_docs,
            current_docs=current_docs,
            stale_meta=stale_meta,
            current_meta=current_meta,
            time_mode=args.time_mode,
            x_qrel=official_linkage.get(x_span.row_index),
            y_qrel=official_linkage.get(y_span.row_index),
            retrieval_config=retrieval_config,
        )
        if record["metadata"]["change_label"] == "changed":
            changed_records.append(record)
        else:
            stable_records.append(record)

    combined: list[dict[str, Any]]
    if args.change_label == "changed":
        combined = changed_records
    elif args.change_label == "stable":
        combined = stable_records
    else:
        combined = changed_records + stable_records

    if common_retriever is not None:
        common_retriever.close()

    write_jsonl(combined, output)
    if output_changed is not None:
        write_jsonl(changed_records, output_changed)
    if output_stable is not None:
        write_jsonl(stable_records, output_stable)

    print(f"Question histories: {len(histories)}")
    print(f"Evidence docs kept: {len(evidence_docs)}")
    print(f"Passages built: {len(passages)}")
    print(f"Retrieval method: {args.retrieval_method}")
    print(f"Official qrels loaded: {len(official_linkage)}")
    print(f"Skipped missing answers at x/y: {skipped_missing_answers}")
    print(f"Skipped by question_type filter: {skipped_filter}")
    print(f"Changed records: {len(changed_records)}")
    print(f"Stable records: {len(stable_records)}")
    print(f"Combined records written: {len(combined)}")
    print(output)
    if output_changed is not None:
        print(output_changed)
    if output_stable is not None:
        print(output_stable)


if __name__ == "__main__":
    main()
