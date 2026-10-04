"""The 300-document language ID sample for hand-labelling (Story 3.6.1, DECISIONS S3-06).

Option C (user): per target language, pages fastText gave that language (split at the
confidence threshold, so the threshold itself can be judged), plus pages Common Crawl's
own detector gave that language but fastText did not (where fastText's misses hide).
Every page belongs to at most one group ("stratum"), so the results can be weighted by
how many pages each group stands for in the crawl.
"""

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

from pyspark import StorageLevel
from pyspark.sql import Column, DataFrame, Row
from pyspark.sql import functions as F

from corpus.io import check_crawl_id, lid_model_path, silver_stage1_path
from corpus.review.label_page import (
    SAMPLE_SEED,
    Card,
    draw,
    group_sizes,
    render_page,
    text_body,
)
from corpus.schemas.silver_stage1 import STAGE1_SCHEMA
from corpus.session import get_session
from corpus.silver.language import LanguageRules, detect_languages

# Pages drawn per group, per target language: 25 + 45 + 30 = 100, so 300 in total.
QUOTAS = {"low": 25, "high": 45, "missed": 30}

# Common Crawl tags pages with ISO 639-3 codes; its first code is its main guess.
CC_CODES = {"en": "eng", "fr": "fra", "ar": "ara"}


def stratum(rules: LanguageRules) -> Column:
    """Which group a page belongs to, e.g. "ar_high", "fr_low", "en_missed"; null if none.

    Needs language_detected, language_conf, cc_language and text. fastText's answer
    decides first: a page fastText gave a target language is in that language's
    low/high group, whatever Common Crawl said. Only pages fastText gave no target
    language can be "missed", by Common Crawl's main guess. Empty pages have nothing
    to label and belong to no group.
    """
    detected, conf = F.col("language_detected"), F.col("language_conf")
    cc_main = F.split("cc_language", ",")[0]
    group: Column = F.lit(None).cast("string")
    for language, threshold in rules.min_conf.items():
        group = (
            F.when((detected == language) & (conf < threshold), F.lit(f"{language}_low"))
            .when(detected == language, F.lit(f"{language}_high"))
            .otherwise(group)
        )
    targets = list(rules.min_conf)
    for language in targets:
        missed = ~detected.isin(targets) | detected.isNull()
        group = F.when(
            group.isNull() & missed & (cc_main == CC_CODES[language]), F.lit(f"{language}_missed")
        ).otherwise(group)
    return F.when(F.col("text") == "", F.lit(None).cast("string")).otherwise(group)


def with_stratum(docs: DataFrame, rules: LanguageRules) -> DataFrame:
    """docs + a `stratum` column."""
    return docs.withColumn("stratum", stratum(rules))


# --- The labelling page ----------------------------------------------------------------------

# The answers the user can give, in button order (keys 1-6). ar = Modern Standard Arabic,
# the corpus's scope; ar-dialect = Arabic, but dialect (e.g. Egyptian, Maghrebi);
# mixed = large parts in two languages.
LABELS = ("en", "fr", "ar", "ar-dialect", "other", "mixed")

# Characters of each page shown: enough to recognise a language, short enough to label
# 300 pages quickly.
SHOWN_CHARACTERS = 3_000


class LabelItem(NamedTuple):
    doc_id: str
    text: str


def render_label_page(crawl_id: str, items: Sequence[LabelItem]) -> str:
    """One self-contained page to label `items`, blind: no URL, no detector's answer.
    Text escaped and cut by label_page.text_body (a crawled "<script>" never runs)."""
    cards = [
        Card(
            item.doc_id,
            text_body(item.text, SHOWN_CHARACTERS),
            f"#{n} - {len(item.text):,} characters",
        )
        for n, item in enumerate(items, start=1)
    ]
    return render_page(
        title=f"Language labelling {crawl_id}",
        intro_html=(
            "<p>Main language of each page: <b>ar</b> = Modern Standard Arabic, "
            "<b>ar-dialect</b> = Arabic dialect, <b>mixed</b> = large parts in two "
            "languages. Keys 1-6, Backspace = back.</p>"
        ),
        storage_key=f"lid-labels-{crawl_id}",
        export_name="lid_labels.csv",
        labels=[(label, label) for label in LABELS],
        cards=cards,
    )


# --- The export job (Spark) --------------------------------------------------------------


def build_record(
    crawl_id: str,
    seed: int,
    rules: LanguageRules,
    sizes: Mapping[str, int],
    picked: Sequence[Row],
) -> dict[str, Any]:
    """What the final numbers need besides the user's labels: each page's group and the
    detectors' answers, the groups' sizes (the weights), and the settings used.
    Ids and numbers only, never page text."""
    return {
        "crawl_id": crawl_id,
        "seed": seed,
        "quotas": dict(QUOTAS),
        "thresholds": dict(rules.min_conf),
        "group_sizes": dict(sorted(sizes.items())),
        "pages": [
            {
                "doc_id": row.doc_id,
                "stratum": row.stratum,
                "language_detected": row.language_detected,
                "language_conf": row.language_conf,
                "cc_language": row.cc_language,
            }
            for row in sorted(picked, key=lambda row: row.doc_id)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crawl-id", required=True)
    parser.add_argument("--out", required=True, help="the labelling page (HTML)")
    parser.add_argument("--record", required=True, help="the sample record (JSON, no text)")
    parser.add_argument("--seed", type=int, default=SAMPLE_SEED)
    args = parser.parse_args()
    check_crawl_id(args.crawl_id)
    rules = LanguageRules()

    started = time.monotonic()
    spark = get_session(f"review lid {args.crawl_id}")
    try:
        spark.sparkContext.addFile(lid_model_path())
        # The full crawl, not the dev subset: the numbers must describe the real corpus.
        docs = spark.read.schema(STAGE1_SCHEMA).parquet(
            silver_stage1_path(args.crawl_id, dev=False)
        )
        labelled = detect_languages(docs.select("doc_id", "cc_language", "text"), rules)
        # Saved once on the workers' disks, without the text: group_sizes and draw both
        # read it, and each read would otherwise run fastText on every page again.
        grouped = (
            with_stratum(labelled, rules)
            .select("doc_id", "stratum", "language_detected", "language_conf", "cc_language")
            .persist(StorageLevel.DISK_ONLY)
        )
        sizes = group_sizes(grouped)
        picks = draw(grouped, QUOTAS, args.seed).join(grouped.drop("stratum"), "doc_id")
        # The text of the ~300 drawn pages only (a broadcast join on doc_id), in a mixed
        # order (a second hash) so the groups do not come in blocks on the page.
        sample = (
            docs.select("doc_id", "text")
            .join(F.broadcast(picks), "doc_id")
            .orderBy(F.xxhash64("doc_id", F.lit(args.seed + 1)), "doc_id")
            .collect()  # ~300 pages: the purpose of a review export, never the corpus
        )
        grouped.unpersist()
    finally:
        spark.stop()

    items = [LabelItem(row.doc_id, row.text) for row in sample]
    Path(args.out).write_text(render_label_page(args.crawl_id, items), encoding="utf-8")
    record = build_record(args.crawl_id, args.seed, rules, sizes, sample)
    Path(args.record).write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")

    drawn: dict[str, int] = {}
    for row in sample:
        drawn[row.stratum] = drawn.get(row.stratum, 0) + 1
    for group in sorted(sizes):
        print(
            f"lid_sample: {group:<10} {sizes[group]:>9,} pages in the crawl, "
            f"{drawn.get(group, 0)} drawn"
        )
    print(f"lid_sample: {len(items)} pages -> {args.out}; record -> {args.record}")
    print(f"lid_sample: {time.monotonic() - started:.1f} s")


if __name__ == "__main__":
    main()
