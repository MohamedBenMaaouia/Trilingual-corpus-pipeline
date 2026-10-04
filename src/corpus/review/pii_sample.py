"""The PII redaction review sample (Story 3.6.2, DECISIONS S3-10).

The unit is a candidate: a piece of the ORIGINAL text (before redaction) that could be
personal contact data. Every email contains "@" and every phone number of the covered
plans has 8+ digits, so these candidates contain all the PII there is:
- "pii_caught": candidates the redactor replaced (measures precision),
- "pii_missed": candidates it left alone (where misses hide; measures recall).
100 of each, from pages in en, fr or ar; weighted by each group's size in the crawl.
The user labels each candidate email / phone / neither, blind to what the redactor did.

    python -m corpus.review.pii_sample --crawl-id CC-MAIN-2026-39 --out ... --record ...
"""

import argparse
import html
import json
import re
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pandas as pd
from pyspark import StorageLevel
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType, LongType, StringType, StructField, StructType

from corpus.io import check_crawl_id, silver_stage1_path, silver_v1_path
from corpus.review.label_page import SAMPLE_SEED, Card, draw, group_sizes, render_page
from corpus.schemas.silver_stage1 import STAGE1_SCHEMA
from corpus.schemas.silver_v1 import SILVER_V1
from corpus.session import get_session
from corpus.silver.pii import find_pii

QUOTAS = {"caught": 100, "missed": 100}
CONTEXT = 150  # characters shown on each side of the candidate
LABELS = [("email", "email"), ("phone", "phone number"), ("neither", "neither")]

# Broader than the redactor on purpose: anything with "@", any run of 8+ digits with
# up to 2 separator characters between them (spaces, dots, dashes, slashes, brackets).
_CANDIDATE = re.compile(r"\S+@\S+|\+?\d(?:[\s().\-/]{0,2}\d){7,}")


def candidates(text: str) -> list[tuple[int, int, bool]]:
    """(start, end, caught) of every candidate in a text; caught = it overlaps a span
    the redactor replaces (silver.pii.find_pii, the redactor's own function)."""
    spans = find_pii(text)
    found = []
    for match in _CANDIDATE.finditer(text):
        start, end = match.start(), match.end()
        caught = any(s.start < end and start < s.end for s in spans)
        found.append((start, end, caught))
    return found


_CANDIDATE_ROWS = StructType(
    [
        StructField("candidate_id", StringType(), False),  # doc_id:start
        StructField("doc_id", StringType(), False),
        StructField("start", LongType(), False),
        StructField("end", LongType(), False),
        StructField("caught", BooleanType(), False),
    ]
)


def _candidate_batches(batches: Iterable[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    """mapInPandas body: (doc_id, text) rows -> one row per candidate. No text out."""
    for batch in batches:
        rows = [
            (f"{doc_id}:{start}", doc_id, start, end, caught)
            for doc_id, text in zip(batch["doc_id"], batch["text"], strict=True)
            for start, end, caught in candidates(text or "")
        ]
        yield pd.DataFrame(rows, columns=_CANDIDATE_ROWS.fieldNames()).astype(
            {
                "candidate_id": object,
                "doc_id": object,
                "start": "int64",
                "end": "int64",
                "caught": bool,
            }
        )


def candidate_table(original: DataFrame) -> DataFrame:
    """(candidate_id, doc_id, start, end, caught, stratum) for (doc_id, text) rows."""
    return (
        original.select("doc_id", "text")
        .mapInPandas(_candidate_batches, schema=_CANDIDATE_ROWS)
        .withColumn(
            "stratum", F.when(F.col("caught"), F.lit("pii_caught")).otherwise(F.lit("pii_missed"))
        )
    )


def card_for(candidate_id: str, text: str, start: int, end: int, number: int) -> Card:
    """The candidate highlighted in its context, every piece escaped."""
    left = text[max(0, start - CONTEXT) : start]
    right = text[end : end + CONTEXT]
    body = (
        "<div class='text' dir='auto'>"
        f"{html.escape(left)}<mark>{html.escape(text[start:end])}</mark>{html.escape(right)}"
        "</div>"
    )
    return Card(candidate_id, body, f"#{number}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crawl-id", required=True)
    parser.add_argument("--out", required=True, help="the labelling page (HTML)")
    parser.add_argument("--record", required=True, help="the sample record (JSON, no text)")
    parser.add_argument("--seed", type=int, default=SAMPLE_SEED)
    args = parser.parse_args()
    check_crawl_id(args.crawl_id)

    started = time.monotonic()
    spark = get_session(f"review pii {args.crawl_id}")
    try:
        # Pages kept in en, fr or ar (any tier) from silver_v1, with their ORIGINAL text
        # from the interim output: the redactor's input, before any placeholder.
        targets = (
            spark.read.schema(SILVER_V1)
            .parquet(silver_v1_path(dev=False))
            .where((F.col("crawl_id") == args.crawl_id) & F.col("language").isin("en", "fr", "ar"))
            .select("doc_id")
        )
        original = spark.read.schema(STAGE1_SCHEMA).parquet(
            silver_stage1_path(args.crawl_id, dev=False)
        )
        pages = original.join(targets, "doc_id")
        table = candidate_table(pages).persist(StorageLevel.DISK_ONLY)  # read twice
        sizes = group_sizes(table)
        picks = draw(table, QUOTAS, args.seed, id_column="candidate_id").join(
            table.drop("stratum"), "candidate_id"
        )
        sample = (
            pages.select("doc_id", "text")
            .join(F.broadcast(picks), "doc_id")
            .orderBy(F.xxhash64("candidate_id", F.lit(args.seed + 1)), "candidate_id")
            .collect()  # 200 rows: a review export, never the corpus
        )
        table.unpersist()
    finally:
        spark.stop()

    cards = [card_for(r.candidate_id, r.text, r.start, r.end, n) for n, r in enumerate(sample, 1)]
    page = render_page(
        title=f"PII review {args.crawl_id}",
        intro_html=(
            "<p>Is the <mark>highlighted</mark> piece an <b>email</b> address, a "
            "<b>phone number</b>, or <b>neither</b> (a date, a price, a code, an id...)? "
            "Judge the highlighted piece only. Keys 1-3, Backspace = back.</p>"
        ),
        storage_key=f"pii-labels-{args.crawl_id}",
        export_name="pii_labels.csv",
        labels=LABELS,
        cards=cards,
    )
    Path(args.out).write_text(page, encoding="utf-8")
    record: dict[str, Any] = {
        "crawl_id": args.crawl_id,
        "seed": args.seed,
        "quotas": QUOTAS,
        "group_sizes": dict(sorted(sizes.items())),
        "candidates": [
            {"candidate_id": r.candidate_id, "stratum": r.stratum, "caught": r.caught}
            for r in sorted(sample, key=lambda r: r.candidate_id)
        ],
    }
    Path(args.record).write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    for group in sorted(sizes):
        drawn = sum(r.stratum == group for r in sample)
        print(f"pii_sample: {group:<11} {sizes[group]:>10,} candidates in the crawl, {drawn} drawn")
    print(f"pii_sample: {len(cards)} cards -> {args.out}; {time.monotonic() - started:.1f} s")


if __name__ == "__main__":
    main()
