"""The rejected-page review sample (Story 3.6.3, DECISIONS S3-10).

200 pages that silver_v1 rejected, in en, fr or ar (pages rejected only for not being
in a target language are out: the question is whether the quality and confidence
filters kill good content). Stratified by language, 67 / 67 / 66, weighted by each
group's size. The user judges each page blind (reasons hidden): good content wrongly
rejected, rightly rejected, or unsure. Filter precision = share rightly rejected.

    python -m corpus.review.rejected_sample --crawl-id CC-MAIN-2026-39 --out ... --record ...
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

from pyspark import StorageLevel
from pyspark.sql import functions as F

from corpus.io import check_crawl_id, silver_v1_path
from corpus.review.label_page import SAMPLE_SEED, Card, draw, group_sizes, render_page, text_body
from corpus.schemas.silver_v1 import SILVER_V1
from corpus.session import get_session

QUOTAS = {"en": 67, "fr": 67, "ar": 66}
SHOWN_CHARACTERS = 3_000
LABELS = [
    ("good", "good content, wrongly rejected"),
    ("junk", "rightly rejected"),
    ("unsure", "unsure"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crawl-id", required=True)
    parser.add_argument("--out", required=True, help="the labelling page (HTML)")
    parser.add_argument("--record", required=True, help="the sample record (JSON, no text)")
    parser.add_argument("--seed", type=int, default=SAMPLE_SEED)
    args = parser.parse_args()
    check_crawl_id(args.crawl_id)

    started = time.monotonic()
    spark = get_session(f"review rejected {args.crawl_id}")
    try:
        rejected = (
            spark.read.schema(SILVER_V1)
            .parquet(silver_v1_path(dev=False))
            .where(
                (F.col("crawl_id") == args.crawl_id)
                & (F.col("quality_tier") == "rejected")
                & F.col("language").isin(*QUOTAS)
            )
            .withColumn("stratum", F.concat(F.lit("rejected_"), F.col("language")))
            .select("doc_id", "stratum", "reject_reasons", "text")
            .persist(StorageLevel.DISK_ONLY)  # read twice: sizes and draw
        )
        sizes = group_sizes(rejected)
        picks = draw(rejected, QUOTAS, args.seed)
        sample = (
            rejected.join(F.broadcast(picks.drop("stratum")), "doc_id")
            .orderBy(F.xxhash64("doc_id", F.lit(args.seed + 1)), "doc_id")
            .collect()  # 200 pages: a review export, never the corpus
        )
        rejected.unpersist()
    finally:
        spark.stop()

    cards = [
        Card(r.doc_id, text_body(r.text, SHOWN_CHARACTERS), f"#{n} - {len(r.text):,} characters")
        for n, r in enumerate(sample, start=1)
    ]
    page = render_page(
        title=f"Rejected pages {args.crawl_id}",
        intro_html=(
            "<p>Each page was <b>rejected</b> by the quality or language-confidence "
            "filters. Would it be <b>good content</b> for a clean corpus (real sentences "
            "worth keeping), or was it <b>rightly rejected</b> (menus, lists, spam, "
            "fragments)? The reasons are hidden on purpose. Text shown with personal "
            "data already redacted. Keys 1-3, Backspace = back.</p>"
        ),
        storage_key=f"rejected-labels-{args.crawl_id}",
        export_name="rejected_labels.csv",
        labels=LABELS,
        cards=cards,
    )
    Path(args.out).write_text(page, encoding="utf-8")
    record: dict[str, Any] = {
        "crawl_id": args.crawl_id,
        "seed": args.seed,
        "quotas": QUOTAS,
        "group_sizes": dict(sorted(sizes.items())),
        "pages": [
            {"doc_id": r.doc_id, "stratum": r.stratum, "reject_reasons": r.reject_reasons}
            for r in sorted(sample, key=lambda r: r.doc_id)
        ],
    }
    Path(args.record).write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    for group in sorted(sizes):
        drawn = sum(r.stratum == group for r in sample)
        print(f"rejected_sample: {group:<12} {sizes[group]:>9,} pages in the crawl, {drawn} drawn")
    print(f"rejected_sample: {len(cards)} cards -> {args.out}; {time.monotonic() - started:.1f} s")


if __name__ == "__main__":
    main()
