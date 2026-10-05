"""corpus.gold.stats: corpus_stats per language x tier (Story 6.4.2, C21)."""

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from pyspark.sql import DataFrame

from corpus.gold.stats import corpus_stats

Batch = Callable[[list[dict[str, Any]]], DataFrame]


def test_counts_lengths_and_dates_per_language_and_tier(gold_v1_batch: Batch) -> None:
    early, late = datetime(2026, 9, 1, tzinfo=UTC), datetime(2026, 9, 20, tzinfo=UTC)
    gold = gold_v1_batch(
        [
            {"doc_id": "a", "char_count": 10, "word_count": 2, "fetch_date": late},
            {"doc_id": "b", "char_count": 31, "word_count": 5, "fetch_date": early},
            {"doc_id": "c", "language": "ar", "quality_tier": "medium", "char_count": 7},
        ]
    )
    rows = [r.asDict() for r in corpus_stats(gold).collect()]
    assert rows == [
        {
            "language": "ar", "quality_tier": "medium", "documents": 1, "chars_total": 7,
            "words_total": 3, "mean_chars": 7.0,
            "first_fetch": datetime(2026, 9, 4, 12, 0), "last_fetch": datetime(2026, 9, 4, 12, 0),
        },
        {
            "language": "en", "quality_tier": "high", "documents": 2, "chars_total": 41,
            "words_total": 7, "mean_chars": 20.5,
            "first_fetch": datetime(2026, 9, 1), "last_fetch": datetime(2026, 9, 20),
        },
    ]  # fmt: skip
