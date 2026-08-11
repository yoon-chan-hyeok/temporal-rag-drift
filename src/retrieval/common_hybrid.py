"""Common BM25-candidate + dense + RRF retrieval for temporal corpora.

CLARK uses this module in the common-retriever arm. Corpus-specific
index builders may expose different metadata columns, but candidate generation,
dense scoring, rank fusion, and document-version deduplication are identical.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.embedding.embed import TextEmbedder


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
    "how",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
}


@dataclass(frozen=True)
class HybridRetrievalConfig:
    """Shared hybrid-ranking configuration."""

    candidate_k: int = 100
    top_k: int = 10
    rrf_k: int = 60
    sparse_weight: float = 1.0
    dense_weight: float = 1.0
    article_dedup: bool = True
    article_dedup_scope: str = "document_version"
    per_query_k: int | None = None
    query_strategy: str = "progressive"


class DenseEmbeddingCache:
    """Dense encoder with a bounded in-memory document/query cache."""

    def __init__(
        self,
        model_name: str,
        *,
        batch_size: int = 32,
        device: str | None = None,
        local_files_only: bool = False,
        allow_hashing_fallback: bool = False,
        query_prefix: str = "Represent this sentence for searching relevant passages: ",
        max_cache_items: int = 50_000,
    ) -> None:
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.max_cache_items = max(0, int(max_cache_items))
        self.embedder = TextEmbedder(
            model_name,
            batch_size=batch_size,
            normalize=True,
            allow_hashing_fallback=allow_hashing_fallback,
            device=device,
            local_files_only=local_files_only,
        )
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def _cache_key(self, kind: str, key: str, text: str) -> str:
        digest = hashlib.sha256()
        digest.update(self.model_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(kind.encode("ascii"))
        digest.update(b"\0")
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(text.encode("utf-8"))
        return digest.hexdigest()

    def _encode_cached(self, items: list[tuple[str, str, str]]) -> np.ndarray:
        if not items:
            return np.zeros((0, 0), dtype=np.float32)

        output: list[np.ndarray | None] = [None] * len(items)
        missing_positions: list[int] = []
        missing_texts: list[str] = []
        missing_keys: list[str] = []

        for position, (kind, key, text) in enumerate(items):
            cache_key = self._cache_key(kind, key, text)
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._cache.move_to_end(cache_key)
                output[position] = cached
                continue
            missing_positions.append(position)
            missing_texts.append(text)
            missing_keys.append(cache_key)

        if missing_texts:
            encoded = self.embedder.encode(missing_texts)
            for position, cache_key, vector in zip(missing_positions, missing_keys, encoded):
                vector = np.asarray(vector, dtype=np.float32)
                output[position] = vector
                if self.max_cache_items > 0:
                    self._cache[cache_key] = vector
                    self._cache.move_to_end(cache_key)
                    while len(self._cache) > self.max_cache_items:
                        self._cache.popitem(last=False)

        return np.stack([vector for vector in output if vector is not None]).astype(np.float32)

    def similarities(
        self,
        query: str,
        documents: list[tuple[str, str]],
    ) -> np.ndarray:
        """Return cosine similarities for ``(cache_key, text)`` documents."""

        if not documents:
            return np.zeros((0,), dtype=np.float32)
        query_text = f"{self.query_prefix}{query}" if self.query_prefix else query
        query_vector = self._encode_cached([("query", query, query_text)])[0]
        document_vectors = self._encode_cached(
            [("document", key, text) for key, text in documents]
        )
        return np.asarray(document_vectors @ query_vector, dtype=np.float32)


def tokenize_query(text: str, max_terms: int = 24) -> list[str]:
    """Tokenize an English factual query for FTS5."""

    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9_'-]*", text.lower())
    terms: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        token = token.strip("'-_")
        if len(token) < 2 or token in STOPWORDS or token in seen:
            continue
        seen.add(token)
        terms.append(token)
        if len(terms) >= max_terms:
            break
    return terms


def quote_fts_term(term: str) -> str:
    """Quote one term so punctuation cannot break FTS5 MATCH syntax."""

    return f'"{term.replace(chr(34), chr(34) + chr(34))}"'


def focus_terms(text: str, max_terms: int = 6) -> list[str]:
    """Extract a likely entity/relation phrase without corpus-specific metadata."""

    raw_tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9_'-]*", text)
    start_index: int | None = None
    for index, token in enumerate(raw_tokens):
        normalized = token.lower().strip("'-_")
        if normalized in STOPWORDS:
            continue
        if token[:1].isupper():
            start_index = index
            break
    if start_index is None:
        return []

    focused: list[str] = []
    for token in raw_tokens[start_index:]:
        normalized = token.lower().strip("'-_")
        if len(normalized) < 2 or normalized in STOPWORDS:
            continue
        focused.append(normalized)
        if len(focused) >= max_terms:
            break
    return focused


def fts_queries(question: str, title: str = "") -> list[str]:
    """Create the same progressively broader FTS queries for every corpus."""

    question_terms = tokenize_query(question)
    title_terms = tokenize_query(title, max_terms=12)
    quoted_question = [quote_fts_term(term) for term in question_terms]
    quoted_title = [quote_fts_term(term) for term in title_terms]
    quoted_focus = [quote_fts_term(term) for term in focus_terms(question)]

    queries: list[str] = []
    if title:
        queries.append(quote_fts_term(title))
    if quoted_title:
        queries.append(" AND ".join(quoted_title[: min(8, len(quoted_title))]))
    for count in (6, 5, 4, 3):
        if len(quoted_focus) >= count:
            queries.append(" AND ".join(quoted_focus[:count]))
    for count in (8, 6, 4, 3):
        if len(quoted_question) >= count:
            queries.append(" AND ".join(quoted_question[:count]))
    combined = quoted_title + quoted_question
    if combined:
        queries.append(" OR ".join(combined[: min(12, len(combined))]))

    deduped: list[str] = []
    for query in queries:
        if query and query not in deduped:
            deduped.append(query)
    return deduped


def single_or_fts_query(question: str, title: str = "") -> str:
    """Return one broad BM25 candidate query for latency-sensitive corpora."""

    terms = tokenize_query(title, max_terms=12) + tokenize_query(question, max_terms=24)
    deduped: list[str] = []
    seen: set[str] = set()
    for term in terms:
        if term in seen:
            continue
        seen.add(term)
        deduped.append(term)
    return " OR ".join(quote_fts_term(term) for term in deduped[:24])


class SQLiteHybridRetriever:
    """Retrieve from an FTS5 index using BM25 candidates, dense scores, and RRF."""

    def __init__(
        self,
        index_path: str | Path,
        dense_cache: DenseEmbeddingCache,
        config: HybridRetrievalConfig,
    ) -> None:
        self.index_path = Path(index_path).resolve()
        self.dense_cache = dense_cache
        self.config = config
        self.connection = sqlite3.connect(
            f"file:{self.index_path.as_posix()}?mode=ro",
            uri=True,
        )
        self.connection.row_factory = sqlite3.Row
        self.columns = {
            str(row["name"])
            for row in self.connection.execute("PRAGMA table_info(chunks)").fetchall()
        }
        if not {"title", "text"}.issubset(self.columns):
            raise ValueError(f"Unsupported chunks schema in {self.index_path}")

        self.timestamp_column = self._first_column("timestamp", "snapshot")
        self.document_id_column = self._first_column("article_id", "page_id")
        self.chunk_index_column = self._first_column("chunk_index")
        self.url_column = self._first_column("url", "archive_url")
        self.source_url_column = self._first_column("source_url")

    def _first_column(self, *names: str) -> str | None:
        for name in names:
            if name in self.columns:
                return name
        return None

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "SQLiteHybridRetriever":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _select_or_empty(self, column: str | None, alias: str) -> str:
        if column is None:
            return f"'' AS {alias}"
        return f"{column} AS {alias}"

    def _sparse_candidates(
        self,
        question: str,
        *,
        cutoff: str | None,
        title: str,
    ) -> list[dict[str, Any]]:
        config = self.config
        candidate_k = max(config.top_k, config.candidate_k)
        per_query_k = max(candidate_k, config.per_query_k or candidate_k)
        aggregated: dict[tuple[str, str, str], dict[str, Any]] = {}

        timestamp_select = self._select_or_empty(self.timestamp_column, "timestamp")
        document_select = self._select_or_empty(self.document_id_column, "document_id")
        chunk_select = self._select_or_empty(self.chunk_index_column, "chunk_index")
        url_select = self._select_or_empty(self.url_column, "url")
        source_url_select = self._select_or_empty(self.source_url_column, "source_url")

        cutoff_clause = ""
        if cutoff is not None:
            if self.timestamp_column is None:
                raise ValueError(
                    f"Index {self.index_path} has no timestamp/snapshot column for cutoff filtering."
                )
            cutoff_clause = f" AND {self.timestamp_column} <= ?"

        sql = f"""
            SELECT
                rowid,
                title,
                text,
                {timestamp_select},
                {document_select},
                {chunk_select},
                {url_select},
                {source_url_select},
                bm25(chunks, 3.0, 1.0) AS bm25_raw
            FROM chunks
            WHERE chunks MATCH ?{cutoff_clause}
            ORDER BY bm25(chunks, 3.0, 1.0) ASC
            LIMIT ?
        """

        if config.query_strategy == "single_or":
            single_query = single_or_fts_query(question, title=title)
            queries = [single_query] if single_query else []
        elif config.query_strategy == "progressive":
            queries = fts_queries(question, title=title)
        else:
            raise ValueError(f"Unknown query_strategy: {config.query_strategy}")

        for query in queries:
            minimum_pool = min(candidate_k, max(config.top_k * 4, 20))
            if " OR " in query and len(aggregated) >= minimum_pool:
                break
            parameters: list[Any] = [query]
            if cutoff is not None:
                parameters.append(cutoff)
            parameters.append(per_query_k)
            rows = self.connection.execute(sql, parameters).fetchall()
            for query_rank, row in enumerate(rows, start=1):
                key = (
                    str(row["timestamp"]),
                    str(row["document_id"] or row["rowid"]),
                    str(row["chunk_index"]),
                )
                query_rrf = 1.0 / (config.rrf_k + query_rank)
                if key not in aggregated:
                    aggregated[key] = {
                        "rowid": int(row["rowid"]),
                        "title": str(row["title"] or ""),
                        "text": str(row["text"] or ""),
                        "timestamp": str(row["timestamp"] or ""),
                        "document_id": str(row["document_id"] or row["rowid"]),
                        "chunk_index": row["chunk_index"],
                        "url": str(row["url"] or ""),
                        "source_url": str(row["source_url"] or ""),
                        "bm25_score": -float(row["bm25_raw"]),
                        "sparse_query_rrf": query_rrf,
                        "matched_queries": 1,
                    }
                else:
                    item = aggregated[key]
                    item["sparse_query_rrf"] += query_rrf
                    item["matched_queries"] += 1
                    item["bm25_score"] = max(
                        float(item["bm25_score"]),
                        -float(row["bm25_raw"]),
                    )

        candidates = sorted(
            aggregated.values(),
            key=lambda row: (
                float(row["sparse_query_rrf"]),
                float(row["bm25_score"]),
                str(row["timestamp"]),
            ),
            reverse=True,
        )
        return candidates[:candidate_k]

    def retrieve(
        self,
        question: str,
        *,
        cutoff: str | None = None,
        title: str = "",
    ) -> list[dict[str, Any]]:
        """Return fused top-k rows, optionally restricted to a temporal cutoff."""

        candidates = self._sparse_candidates(question, cutoff=cutoff, title=title)
        if not candidates:
            return []

        for sparse_rank, candidate in enumerate(candidates, start=1):
            candidate["sparse_rank"] = sparse_rank

        dense_documents = [
            (
                "|".join(
                    [
                        str(candidate["timestamp"]),
                        str(candidate["document_id"]),
                        str(candidate["chunk_index"]),
                    ]
                ),
                f"{candidate['title']}\n{candidate['text']}".strip(),
            )
            for candidate in candidates
        ]
        dense_scores = self.dense_cache.similarities(question, dense_documents)
        dense_order = np.argsort(-dense_scores, kind="stable")
        dense_ranks = np.empty(len(candidates), dtype=np.int64)
        dense_ranks[dense_order] = np.arange(1, len(candidates) + 1)

        config = self.config
        for index, candidate in enumerate(candidates):
            sparse_rank = int(candidate["sparse_rank"])
            dense_rank = int(dense_ranks[index])
            candidate["dense_score"] = float(dense_scores[index])
            candidate["dense_rank"] = dense_rank
            candidate["rrf_score"] = (
                config.sparse_weight / (config.rrf_k + sparse_rank)
                + config.dense_weight / (config.rrf_k + dense_rank)
            )

        fused = sorted(
            candidates,
            key=lambda row: (
                float(row["rrf_score"]),
                float(row["dense_score"]),
                float(row["sparse_query_rrf"]),
            ),
            reverse=True,
        )

        selected: list[dict[str, Any]] = []
        seen_documents: set[tuple[str, ...]] = set()
        for candidate in fused:
            if config.article_dedup_scope == "document":
                document_key = (str(candidate["document_id"]),)
            elif config.article_dedup_scope == "document_version":
                document_key = (
                    str(candidate["timestamp"]),
                    str(candidate["document_id"]),
                )
            else:
                raise ValueError(
                    "article_dedup_scope must be 'document' or 'document_version', "
                    f"got {config.article_dedup_scope!r}"
                )
            if config.article_dedup and document_key in seen_documents:
                continue
            seen_documents.add(document_key)
            selected.append(candidate)
            if len(selected) >= config.top_k:
                break

        for rank, candidate in enumerate(selected, start=1):
            candidate["rank"] = rank
            candidate["score"] = float(candidate["rrf_score"])
            candidate["retrieval_method"] = "common_bm25_dense_rrf"
        return selected


def format_retrieved_doc(row: dict[str, Any], rank: int | None = None) -> str:
    """Format one shared-retriever row as generator context."""

    display_rank = int(rank if rank is not None else row.get("rank", 0))
    lines = [
        f"[retrieved rank {display_rank}]",
        f"Timestamp: {row.get('timestamp', '')}",
        f"Title: {row.get('title', '')}",
    ]
    source_url = str(row.get("source_url") or row.get("url") or "")
    if source_url:
        lines.append(f"Source URL: {source_url}")
    lines.extend(
        [
            f"Document ID: {row.get('document_id', '')}",
            f"Chunk: {row.get('chunk_index', '')}",
            f"RRF score: {float(row.get('rrf_score', row.get('score', 0.0))):.8f}",
            f"Sparse rank: {row.get('sparse_rank', '')}",
            f"Dense rank: {row.get('dense_rank', '')}",
            str(row.get("text") or ""),
        ]
    )
    return "\n".join(lines)
