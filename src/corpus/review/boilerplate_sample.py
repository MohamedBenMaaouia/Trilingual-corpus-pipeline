"""The 50-document boilerplate review (Sprint 2 DoD, DECISIONS S2-08).

For each sampled document: every line of the text before boilerplate removal, labelled
with the rule that removed it (A, B, C, "domain") or kept. Rendered as a local HTML
page for the user to read. Never published: the text comes from third-party websites.
"""

import argparse
import html
from functools import reduce
from pathlib import Path
from typing import NamedTuple

import xxhash
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from corpus.bronze.control import SegmentControl
from corpus.db import connect
from corpus.io import check_crawl_id
from corpus.jobs.run_silver import DEV_SEGMENTS
from corpus.session import get_session
from corpus.silver.boilerplate import (
    DomainRules,
    LineRules,
    cap_entries,
    domain_boilerplate,
    judge_line,
    remove_domain_boilerplate,
)
from corpus.silver.parse import parse_files

# Which documents to show (user, S2-08): Arabic is under 1% of documents, so a purely
# random 50 would show almost none. 2 per language are documents that end up emptied.
LANGUAGE_QUOTAS = {"eng": 20, "fra": 15, "ara": 15}
EMPTIED_PER_LANGUAGE = 2
SAMPLE_SEED = 42  # fixed: the same 50 documents on every run

# Rules that remove nothing: gives the "before" text (after normalize_common only).
KEEP_EVERYTHING = LineRules(min_words=0, min_words_without_end=0, max_non_letter_share=1.0)

# Spark's xxhash64(string) is XXH64 of the UTF-8 bytes with seed 42, returned as a signed
# 64-bit number (Spark's LongType). Reproduced here to label the domain rule's lines.
SPARK_HASH_SEED = 42


def spark_xxhash64(text: str) -> int:
    """The value Spark's F.xxhash64(text) gives, computed in plain Python."""
    value = xxhash.xxh64_intdigest(text.encode("utf-8"), seed=SPARK_HASH_SEED)
    return value - 2**64 if value >= 2**63 else value  # unsigned -> signed, like LongType


class ReviewLine(NamedTuple):
    text: str
    removed_by: str | None  # "A", "B", "C", "domain", or None if kept


class ReviewDocument(NamedTuple):
    doc_id: str
    url: str
    domain: str
    language: str  # Common Crawl's primary tag: eng, fra, ara
    lines: list[ReviewLine]


def label_lines(before: str, domain_hashes: frozenset[int], rules: LineRules) -> list[ReviewLine]:
    """Label each line of a normalized text the way the pipeline treats it.

    Same order as the pipeline: the per-line rules first (judge_line), then the domain
    rule on the lines they kept. Blank lines are skipped, as clean_lines does.
    """
    labelled: list[ReviewLine] = []
    for line in before.split("\n"):
        if not line:
            continue
        rule = judge_line(line, rules)
        if rule is None and spark_xxhash64(line) in domain_hashes:
            rule = "domain"
        labelled.append(ReviewLine(line, rule))
    return labelled


def kept_text(lines: list[ReviewLine]) -> str:
    """The text the pipeline keeps, rebuilt from the labels (checked against Spark's)."""
    return "\n".join(line.text for line in lines if line.removed_by is None)


_CSS = """
body { font: 15px/1.5 system-ui, sans-serif; max-width: 60rem; margin: 2rem auto;
       padding: 0 1rem; color: #1a1a1a; background: #fafafa; }
h1 { font-size: 1.4rem; } h2 { font-size: 1.05rem; margin: 0 0 .3rem; }
.doc { background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 1rem;
       margin: 1.5rem 0; }
.meta { color: #555; font-size: .85rem; word-break: break-all; margin-bottom: .6rem; }
.line { padding: 1px 4px; border-left: 3px solid #2e7d32; margin: 1px 0; }
.removed { border-left-color: #bbb; color: #888; text-decoration: line-through; }
.tag { display: inline-block; min-width: 3.6rem; font: 11px monospace; color: #fff;
       background: #777; border-radius: 3px; padding: 0 4px; margin-right: 6px;
       text-decoration: none; }
.tag.A { background: #1565c0; } .tag.B { background: #6a1b9a; }
.tag.C { background: #ef6c00; } .tag.domain { background: #c62828; }
.tag.kept { background: #2e7d32; }
"""


def render_html(crawl_id: str, docs: list[ReviewDocument]) -> str:
    """One self-contained page. Every piece of document text is HTML-escaped: it comes
    from untrusted websites, and a "<script>" in it must show as text, never run."""
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>Boilerplate review {html.escape(crawl_id)}</title>",
        f"<style>{_CSS}</style></head><body>",
        f"<h1>Boilerplate review: {len(docs)} documents, {html.escape(crawl_id)}</h1>",
        "<p>Green = kept. Struck through = removed, tagged with the rule: "
        "<b>A</b> fewer than 4 words, <b>B</b> no sentence end and fewer than 10 words, "
        "<b>C</b> mostly non-letters, <b>domain</b> repeated across the site.</p>",
    ]
    for number, doc in enumerate(docs, start=1):
        removed = sum(line.removed_by is not None for line in doc.lines)
        verdict = "EMPTIED" if removed == len(doc.lines) else f"{removed}/{len(doc.lines)} removed"
        parts.append("<div class='doc'>")
        parts.append(
            f"<h2>{number}. [{html.escape(doc.language)}] {html.escape(doc.domain)}"
            f" &middot; {verdict}</h2>"
        )
        parts.append(f"<div class='meta'>{html.escape(doc.url)}<br>{html.escape(doc.doc_id)}</div>")
        for line in doc.lines:
            tag = line.removed_by or "kept"
            css = "line" if line.removed_by is None else "line removed"
            parts.append(
                f"<div class='{css}' dir='auto'><span class='tag {tag}'>{tag}</span>"
                f"{html.escape(line.text)}</div>"
            )
        parts.append("</div>")
    parts.append("</body></html>")
    return "\n".join(parts)


# --- The export job (Spark) --------------------------------------------------------------


def pick_sample(docs: DataFrame, seed: int) -> DataFrame:
    """LANGUAGE_QUOTAS documents per language, EMPTIED_PER_LANGUAGE of them emptied.

    Deterministic without randomness: documents are ordered by xxhash64(doc_id, seed)
    (CLAUDE.md section 8: never rand()), doc_id breaking the (unlikely) ties.
    """
    order = [F.xxhash64("doc_id", F.lit(seed)), F.col("doc_id")]
    picks = []
    for language, quota in LANGUAGE_QUOTAS.items():
        pool = docs.where(F.col("language") == language)
        picks.append(pool.where("emptied").orderBy(*order).limit(EMPTIED_PER_LANGUAGE))
        others = pool.where(~F.col("emptied")).orderBy(*order)
        picks.append(others.limit(quota - EMPTIED_PER_LANGUAGE))
    return reduce(DataFrame.unionByName, picks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crawl-id", required=True)
    parser.add_argument("--out", required=True, help="where to write the HTML page")
    parser.add_argument("--seed", type=int, default=SAMPLE_SEED)
    args = parser.parse_args()
    check_crawl_id(args.crawl_id)

    with connect() as conn:  # the dev segments, exactly as run_silver --dev reads them
        files = SegmentControl(conn).complete_files(args.crawl_id)[:DEV_SEGMENTS]
    line_rules, domain_rules = LineRules(), DomainRules()

    spark = get_session(f"review boilerplate {args.crawl_id}")
    try:
        raw = spark.read.format("binaryFile").load([key for _, key in files])
        documents = F.col("exception").isNull()
        before = (
            parse_files(raw, KEEP_EVERYTHING)
            .where(documents)
            .select(
                "doc_id",
                "url",
                "domain",
                F.split("cc_language", ",")[0].alias("language"),
                F.col("text").alias("before"),
            )
        )
        # The real pipeline, as run_silver runs it.
        real = parse_files(raw, line_rules).where(documents)
        boilerplate = domain_boilerplate(real.select("domain", "text"), domain_rules).persist()
        after = remove_domain_boilerplate(real, boilerplate, domain_rules).select(
            "doc_id", F.col("text").alias("after")
        )
        # Persisted: pick_sample runs 6 queries on it (2 per language). Without this,
        # Spark recomputes both parses for each one: the first run read bronze 11 times
        # and took 308 s (S2-08).
        joined = before.join(after, "doc_id").withColumn("emptied", F.col("after") == "")
        joined = joined.persist()
        # 50 documents to the driver: the purpose of a review export (never the corpus).
        sample = pick_sample(joined, args.seed).collect()
        domains = sorted({row.domain for row in sample})
        entries = (
            cap_entries(boilerplate, domain_rules)
            .where(F.col("domain").isin(domains))
            .select("domain", "line_hash")
            .collect()
        )
    finally:
        spark.stop()

    hashes: dict[str, set[int]] = {}
    for entry in entries:
        hashes.setdefault(entry.domain, set()).add(entry.line_hash)
    review, mismatches = [], []
    for row in sample:
        lines = label_lines(row.before, frozenset(hashes.get(row.domain, set())), line_rules)
        if kept_text(lines) != row.after:  # the labels must rebuild Spark's exact output
            mismatches.append(row.doc_id)
        review.append(ReviewDocument(row.doc_id, row.url, row.domain, row.language, lines))
    if mismatches:
        raise SystemExit(f"labels disagree with the pipeline for {len(mismatches)} documents")

    Path(args.out).write_text(render_html(args.crawl_id, review), encoding="utf-8")
    per_language = {lang: sum(d.language == lang for d in review) for lang in LANGUAGE_QUOTAS}
    print(f"review: {len(review)} documents {per_language}, labels match Spark, -> {args.out}")


if __name__ == "__main__":
    main()
