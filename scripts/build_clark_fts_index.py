"""Build the CLARK-News SQLite FTS5 index used by the common retriever."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.text import compact_whitespace, stable_text_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a temporal FTS5 index from CLARK external_sources.json.")
    parser.add_argument(
        "--evidence-file",
        default="data/external/clark/external_sources.json",
    )
    parser.add_argument(
        "--output",
        default="outputs/indexes/clark_news_common.sqlite",
    )
    parser.add_argument("--chunk-words", type=int, default=180)
    parser.add_argument("--chunk-overlap", type=int, default=40)
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (ROOT / candidate).resolve()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_title(text: str, max_chars: int = 180) -> str:
    for line in text.splitlines():
        line = compact_whitespace(line)
        if len(line) >= 8:
            return line[:max_chars]
    return compact_whitespace(text)[:max_chars]


def chunk_text(
    text: str,
    *,
    chunk_words: int,
    overlap: int,
    min_chars: int,
) -> Iterator[tuple[int, str]]:
    words = compact_whitespace(text).split()
    if not words:
        return
    step = max(1, chunk_words - overlap)
    chunk_index = 0
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_words]).strip()
        if len(chunk) >= min_chars:
            yield chunk_index, chunk
            chunk_index += 1
        if start + chunk_words >= len(words):
            break


def iter_article_versions(path: Path) -> Iterator[dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("CLARK external_sources.json must be an object keyed by source URL.")

    seen_versions: set[tuple[str, str]] = set()
    for source_url, versions in data.items():
        if not isinstance(versions, dict):
            continue
        for timestamp_key, payload in versions.items():
            if not isinstance(payload, dict):
                continue
            timestamp = str(payload.get("source_timestamp") or timestamp_key).strip()
            archive_url = str(payload.get("archive_url") or source_url).strip()
            text = str(payload.get("source_text") or "").strip()
            if not timestamp or len(text) < 80:
                continue
            version_key = (archive_url, timestamp)
            if version_key in seen_versions:
                continue
            seen_versions.add(version_key)
            yield {
                "source_url": str(source_url).strip(),
                "url": archive_url,
                "timestamp": timestamp,
                "title": infer_title(text),
                "text": text,
            }


def init_index(path: Path, force: bool) -> sqlite3.Connection:
    if path.exists():
        if not force:
            raise FileExistsError(f"Output already exists. Use --force to replace it: {path}")
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        """
        CREATE VIRTUAL TABLE chunks USING fts5(
            title,
            text,
            timestamp UNINDEXED,
            article_id UNINDEXED,
            chunk_index UNINDEXED,
            url UNINDEXED,
            source_url UNINDEXED,
            tokenize='unicode61 remove_diacritics 2'
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE index_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    connection.executemany(
        "INSERT INTO index_meta(key, value) VALUES (?, ?)",
        [
            ("schema_version", "2"),
            ("source_type", "clark_news"),
            ("chunking", "word_window"),
        ],
    )
    connection.commit()
    return connection


def build_index(
    evidence_file: Path,
    output: Path,
    *,
    chunk_words: int,
    chunk_overlap: int,
    min_chars: int,
    batch_size: int,
    force: bool,
) -> dict[str, Any]:
    connection = init_index(output, force=force)
    batch: list[tuple[str, str, str, str, int, str, str]] = []
    article_versions = 0
    chunks = 0
    try:
        for article in iter_article_versions(evidence_file):
            article_versions += 1
            article_id = stable_text_hash(
                f"{article['source_url']}|{article['url']}|{article['timestamp']}"
            )
            for chunk_index, chunk in chunk_text(
                article["text"],
                chunk_words=chunk_words,
                overlap=chunk_overlap,
                min_chars=min_chars,
            ):
                batch.append(
                    (
                        article["title"],
                        chunk,
                        article["timestamp"],
                        article_id,
                        chunk_index,
                        article["url"],
                        article["source_url"],
                    )
                )
                chunks += 1
                if len(batch) >= batch_size:
                    connection.executemany(
                        """
                        INSERT INTO chunks(
                            title, text, timestamp, article_id, chunk_index, url, source_url
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        batch,
                    )
                    connection.commit()
                    batch.clear()
        if batch:
            connection.executemany(
                """
                INSERT INTO chunks(
                    title, text, timestamp, article_id, chunk_index, url, source_url
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                batch,
            )
            connection.commit()
        connection.executemany(
            "INSERT OR REPLACE INTO index_meta(key, value) VALUES (?, ?)",
            [
                ("evidence_file", str(evidence_file)),
                ("evidence_sha256", file_sha256(evidence_file)),
                ("article_versions", str(article_versions)),
                ("chunks", str(chunks)),
                ("chunk_words", str(chunk_words)),
                ("chunk_overlap", str(chunk_overlap)),
            ],
        )
        connection.commit()
    finally:
        connection.close()
    return {
        "evidence_file": str(evidence_file),
        "output": str(output),
        "article_versions": article_versions,
        "chunks": chunks,
        "chunk_words": chunk_words,
        "chunk_overlap": chunk_overlap,
    }


def main() -> None:
    args = parse_args()
    result = build_index(
        resolve(args.evidence_file),
        resolve(args.output),
        chunk_words=int(args.chunk_words),
        chunk_overlap=int(args.chunk_overlap),
        min_chars=int(args.min_chars),
        batch_size=int(args.batch_size),
        force=bool(args.force),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
