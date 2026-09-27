"""corpus.review.boilerplate_sample: the pure pieces of the 50-document review (S2-08)."""

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from corpus.review.boilerplate_sample import (
    ReviewDocument,
    ReviewLine,
    kept_text,
    label_lines,
    render_html,
    spark_xxhash64,
)
from corpus.silver.boilerplate import LineRules

RULES = LineRules()
FOOTER = "Subscribe to our weekly newsletter today."
SENTENCE = "The committee met on Friday and approved the new budget."


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Home",
        FOOTER,
        "Est-ce que vous venez ce soir ?",
        "آخر الأخبار من فريقنا اليوم",
        "emoji \N{GRINNING FACE} and accents: caf\N{LATIN SMALL LETTER E WITH ACUTE}",
    ],
)
def test_python_hash_equals_spark_xxhash64(spark: SparkSession, text: str) -> None:
    """If this drifted, every "domain" label in the review would silently be wrong."""
    row = spark.range(1).select(F.xxhash64(F.lit(text)).alias("h")).first()
    assert row is not None
    assert spark_xxhash64(text) == row.h


def test_lines_are_labelled_like_the_pipeline_treats_them() -> None:
    before = "\n".join(["Home", "Latest news from our team", "", SENTENCE, FOOTER])
    labels = label_lines(before, frozenset({spark_xxhash64(FOOTER)}), RULES)
    assert labels == [
        ReviewLine("Home", "A"),
        ReviewLine("Latest news from our team", "B"),
        ReviewLine(SENTENCE, None),  # the blank line is skipped, like clean_lines
        ReviewLine(FOOTER, "domain"),
    ]


def test_a_line_caught_by_a_per_line_rule_is_never_labelled_domain() -> None:
    """Same order as the pipeline: the domain rule only sees lines the others kept."""
    labels = label_lines("Home", frozenset({spark_xxhash64("Home")}), RULES)
    assert labels == [ReviewLine("Home", "A")]


def test_kept_text_rebuilds_what_the_pipeline_keeps() -> None:
    labels = [ReviewLine("Home", "A"), ReviewLine(SENTENCE, None), ReviewLine(FOOTER, "domain")]
    assert kept_text(labels) == SENTENCE


def test_document_text_is_escaped_so_it_can_never_run() -> None:
    """Crawl text is untrusted: a script tag in a page must show as text in the review."""
    hostile = "<script>alert('x')</script> & <b>bold</b>"
    doc = ReviewDocument(
        "id-1", "http://e.com/?a=1&b=<x>", "e.com", "eng", [ReviewLine(hostile, None)]
    )
    page = render_html("CC-MAIN-2026-39", [doc])
    assert "<script>" not in page
    assert "&lt;script&gt;alert(&#x27;x&#x27;)&lt;/script&gt; &amp; &lt;b&gt;" in page
    assert "http://e.com/?a=1&amp;b=&lt;x&gt;" in page


def test_a_fully_removed_document_is_flagged_emptied() -> None:
    doc = ReviewDocument("id-2", "http://e.com/", "e.com", "fra", [ReviewLine("Accueil", "A")])
    assert "EMPTIED" in render_html("CC-MAIN-2026-39", [doc])
