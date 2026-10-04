"""dedup.exact: the exact hash and each text's kept page (Story 4.1, DECISIONS S4-02)."""

from datetime import UTC, datetime

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from corpus.dedup.exact import exact_hash, exact_representatives

DAY = [datetime(2026, 9, d, tzinfo=UTC) for d in range(1, 5)]
# "book" with its vowel marks (fatha on kaf and teh, damma on beh) and without (D6).
BOOK_VOWELLED = (
    "\N{ARABIC LETTER KAF}\N{ARABIC FATHA}\N{ARABIC LETTER TEH}\N{ARABIC FATHA}"
    "\N{ARABIC LETTER BEH}\N{ARABIC DAMMA}"
)
BOOK = "\N{ARABIC LETTER KAF}\N{ARABIC LETTER TEH}\N{ARABIC LETTER BEH}"


def test_the_hash_ignores_arabic_spelling_only(spark: SparkSession) -> None:
    rows = spark.createDataFrame(
        [
            ("a", f"the {BOOK_VOWELLED} is here"),
            ("b", f"the {BOOK} is here"),
            ("c", "the book is here"),
            ("d", "the book is here"),
            ("e", "The book is here"),
        ],
        "doc_id string, text string",
    ).select("doc_id", exact_hash(F.col("text")).alias("h"))
    hashes = {r.doc_id: r.h for r in rows.collect()}
    assert hashes["a"] == hashes["b"]  # vowel marks folded by the matching key
    assert hashes["c"] == hashes["d"]
    assert hashes["c"] != hashes["e"]  # exact means exact: case is a near-dup matter
    assert hashes["a"] != hashes["c"]
    assert all(isinstance(h, int) for h in hashes.values())  # a signed long, never null


def test_each_text_keeps_its_best_page(spark: SparkSession) -> None:
    """Highest score first, then earliest fetch, then smallest doc_id (C13)."""
    docs = spark.createDataFrame(
        [
            ("late", 1, 0.9, DAY[2]),
            ("early", 1, 0.9, DAY[0]),  # same text and score: the earliest copy wins
            ("better", 2, 0.95, DAY[3]),
            ("worse", 2, 0.5, DAY[0]),  # the higher score wins even when fetched later
            ("tie-b", 3, 0.7, DAY[1]),
            ("tie-a", 3, 0.7, DAY[1]),  # full tie: the smallest doc_id
            ("alone", 4, 0.1, DAY[0]),
        ],
        "doc_id string, exact_hash long, quality_score double, fetch_date timestamp",
    )
    reps = {r.doc_id: r.exact_rep for r in exact_representatives(docs).collect()}
    assert reps == {
        "late": "early",
        "early": "early",
        "better": "better",
        "worse": "better",
        "tie-b": "tie-a",
        "tie-a": "tie-a",
        "alone": "alone",
    }
