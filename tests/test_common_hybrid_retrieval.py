from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.retrieval.common_hybrid import (
    DenseEmbeddingCache,
    HybridRetrievalConfig,
    SQLiteHybridRetriever,
    focus_terms,
    single_or_fts_query,
)


class CommonHybridRetrievalTest(unittest.TestCase):
    def build_index(self, path: Path) -> None:
        connection = sqlite3.connect(path)
        connection.execute(
            """
            CREATE VIRTUAL TABLE chunks USING fts5(
                title,
                text,
                timestamp UNINDEXED,
                article_id UNINDEXED,
                chunk_index UNINDEXED,
                url UNINDEXED,
                source_url UNINDEXED
            )
            """
        )
        connection.executemany(
            """
            INSERT INTO chunks(
                title, text, timestamp, article_id, chunk_index, url, source_url
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "Acme leadership",
                    "Alice became the chief executive officer of Acme in 2023.",
                    "2023-06-01T00:00:00+00:00",
                    "acme-2023",
                    0,
                    "https://archive.example/acme-2023",
                    "https://example.com/acme-2023",
                ),
                (
                    "Acme leadership",
                    "The company also announced a new product.",
                    "2023-06-01T00:00:00+00:00",
                    "acme-2023",
                    1,
                    "https://archive.example/acme-2023",
                    "https://example.com/acme-2023",
                ),
                (
                    "Acme leadership",
                    "Bob became the chief executive officer of Acme in 2024.",
                    "2024-06-01T00:00:00+00:00",
                    "acme-2024",
                    0,
                    "https://archive.example/acme-2024",
                    "https://example.com/acme-2024",
                ),
                (
                    "Acme leadership follow-up",
                    "The Acme leadership article was updated later in 2024.",
                    "2024-07-01T00:00:00+00:00",
                    "acme-2023",
                    2,
                    "https://archive.example/acme-2023",
                    "https://example.com/acme-2023",
                ),
                (
                    "Unrelated sports",
                    "A local team won a match.",
                    "2024-01-01T00:00:00+00:00",
                    "sports",
                    0,
                    "https://archive.example/sports",
                    "https://example.com/sports",
                ),
            ],
        )
        connection.commit()
        connection.close()

    def test_focus_terms_extracts_entity_phrase(self) -> None:
        self.assertEqual(
            focus_terms("Who is the CEO of Acme Corporation now?")[:2],
            ["ceo", "acme"],
        )

    def test_single_or_query_deduplicates_title_and_question_terms(self) -> None:
        query = single_or_fts_query(
            "What is the status of Acme and Beta?",
            title="Acme Beta",
        )
        self.assertEqual(query.count('"acme"'), 1)
        self.assertIn('"beta"', query)

    def test_temporal_cutoff_and_document_dedup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = Path(temp_dir) / "test.sqlite"
            self.build_index(index_path)
            dense_cache = DenseEmbeddingCache(
                "hashing",
                allow_hashing_fallback=True,
                query_prefix="",
            )
            retriever = SQLiteHybridRetriever(
                index_path,
                dense_cache,
                HybridRetrievalConfig(
                    candidate_k=20,
                    top_k=5,
                    article_dedup=True,
                ),
            )
            try:
                old_rows = retriever.retrieve(
                    "Who is the chief executive officer of Acme?",
                    cutoff="2023-12-31T23:59:59+00:00",
                )
                new_rows = retriever.retrieve(
                    "Who is the chief executive officer of Acme?",
                    cutoff="2024-12-31T23:59:59+00:00",
                )
            finally:
                retriever.close()

            self.assertTrue(any("Alice" in row["text"] for row in old_rows))
            self.assertFalse(any("Bob" in row["text"] for row in old_rows))
            self.assertTrue(any("Bob" in row["text"] for row in new_rows))
            old_versions = [
                (row["timestamp"], row["document_id"])
                for row in new_rows
            ]
            self.assertEqual(len(old_versions), len(set(old_versions)))

    def test_document_scope_keeps_one_chunk_per_article_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = Path(temp_dir) / "test.sqlite"
            self.build_index(index_path)
            dense_cache = DenseEmbeddingCache(
                "hashing",
                allow_hashing_fallback=True,
                query_prefix="",
            )
            retriever = SQLiteHybridRetriever(
                index_path,
                dense_cache,
                HybridRetrievalConfig(
                    candidate_k=20,
                    top_k=10,
                    article_dedup=True,
                    article_dedup_scope="document",
                ),
            )
            try:
                rows = retriever.retrieve(
                    "What happened in the Acme leadership article?",
                    cutoff="2024-12-31T23:59:59+00:00",
                )
            finally:
                retriever.close()

            document_ids = [row["document_id"] for row in rows]
            self.assertEqual(len(document_ids), len(set(document_ids)))
            self.assertEqual(document_ids.count("acme-2023"), 1)


if __name__ == "__main__":
    unittest.main()
