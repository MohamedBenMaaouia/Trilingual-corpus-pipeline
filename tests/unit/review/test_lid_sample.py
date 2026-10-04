"""corpus.review.lid_sample: the language ID labelling sample (Story 3.6.1, S3-06)."""

import json

from pyspark.sql import DataFrame, SparkSession

from corpus.review.label_page import draw, group_sizes
from corpus.review.lid_sample import (
    LABELS,
    SHOWN_CHARACTERS,
    LabelItem,
    build_record,
    render_label_page,
    with_stratum,
)
from corpus.silver.language import LanguageRules

COLUMNS = (
    "doc_id string, language_detected string, language_conf double, cc_language string, text string"
)


def test_each_page_gets_the_right_group(spark: SparkSession) -> None:
    rows = [
        ("ar-sure", "ar", 0.97, "ara", "t"),
        ("ar-unsure", "ar", 0.40, "ara", "t"),
        ("ar-at-threshold", "ar", 0.65, "ara", "t"),  # 0.65 is kept, so "high"
        ("fr-but-cc-eng", "fr", 0.90, "eng", "t"),  # fastText decides first
        ("arz-cc-ara", "arz", 0.80, "ara", "t"),  # Egyptian per fastText: an Arabic miss?
        ("de-cc-fra-eng", "de", 0.70, "fra,eng", "t"),  # Common Crawl's main guess: fra
        ("de-cc-deu", "de", 0.90, "deu", "t"),  # neither detector says en/fr/ar
        ("empty", None, 0.0, "eng", ""),  # nothing to label
        ("no-cc-tag", "es", 0.90, None, "t"),
    ]
    docs = spark.createDataFrame(rows, COLUMNS)

    groups = {
        r.doc_id: r.stratum
        for r in with_stratum(docs, LanguageRules()).collect()  # 9 test rows
    }

    assert groups == {
        "ar-sure": "ar_high",
        "ar-unsure": "ar_low",
        "ar-at-threshold": "ar_high",
        "fr-but-cc-eng": "fr_high",
        "arz-cc-ara": "ar_missed",
        "de-cc-fra-eng": "fr_missed",
        "de-cc-deu": None,
        "empty": None,
        "no-cc-tag": None,
    }


def test_groups_follow_each_languages_threshold(spark: SparkSession) -> None:
    docs = spark.createDataFrame([("x", "ar", 0.7, "ara", "t")], COLUMNS)
    strict = LanguageRules(min_conf={"en": 0.65, "fr": 0.65, "ar": 0.8})
    assert with_stratum(docs, strict).first().stratum == "ar_low"  # type: ignore[union-attr]


# Drawing and group sizes -----------------------------------------------------------------

SMALL = {"low": 2, "high": 3, "missed": 1}  # small quotas, so the test data stays small


def grouped(spark: SparkSession) -> DataFrame:
    """10 en_high pages, 1 ar_low page, 4 fr_missed pages, 2 pages in no group."""
    rows = (
        [(f"en-{i}", "en_high") for i in range(10)]
        + [("ar-0", "ar_low")]
        + [(f"fr-{i}", "fr_missed") for i in range(4)]
        + [("none-0", None), ("none-1", None)]
    )
    return spark.createDataFrame(rows, "doc_id string, stratum string")


def test_draw_takes_each_groups_quota(spark: SparkSession) -> None:
    picked = [r.stratum for r in draw(grouped(spark), SMALL).collect()]
    assert sorted(picked) == ["ar_low"] + ["en_high"] * 3 + ["fr_missed"]


def test_a_short_group_gives_all_its_pages(spark: SparkSession) -> None:
    picked = {r.doc_id for r in draw(grouped(spark), SMALL).collect()}
    assert "ar-0" in picked  # quota 2, only 1 page: it is taken


def test_draw_is_deterministic_and_depends_on_the_seed(spark: SparkSession) -> None:
    def ids(seed: int) -> list[str]:
        return sorted(r.doc_id for r in draw(grouped(spark), SMALL, seed).collect())

    assert ids(42) == ids(42)  # same pages on every run
    assert ids(42) != ids(7)  # the hash order really decides (10 en_high pages, 3 taken)


def test_group_sizes_count_every_page_of_each_group(spark: SparkSession) -> None:
    assert group_sizes(grouped(spark)) == {"en_high": 10, "ar_low": 1, "fr_missed": 4}


# The labelling page ----------------------------------------------------------------------


def test_page_text_is_escaped_never_run() -> None:
    page = render_label_page("CC-MAIN-2026-39", [LabelItem("d1", "<script>alert(1)</script> & co")])
    assert "<script>alert(1)" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt; &amp; co" in page


def test_page_has_one_card_per_item_and_every_label() -> None:
    items = [LabelItem(f"d{i}", "text") for i in range(3)]
    page = render_label_page("CC-MAIN-2026-39", items)
    assert page.count("class='card'") == 3
    assert all(f"data-id='d{i}'" in page for i in range(3))
    assert all(f"data-label='{label}'" in page for label in LABELS)


def test_long_pages_are_cut_and_say_so() -> None:
    page = render_label_page("CC-MAIN-2026-39", [LabelItem("d1", "a" * (SHOWN_CHARACTERS + 500))])
    assert "a" * SHOWN_CHARACTERS in page and "a" * (SHOWN_CHARACTERS + 1) not in page
    assert "500 more characters not shown" in page


def test_the_page_is_blind() -> None:
    # Only ids and text reach the page: nothing that hints at a detector's answer.
    assert LabelItem._fields == ("doc_id", "text")


def test_the_record_holds_ids_and_numbers_never_text(spark: SparkSession) -> None:
    picked = spark.createDataFrame(
        [
            ("d2", "ar_missed", "arz", 0.8, "ara", "secret page text"),
            ("d1", "en_high", "en", 0.9, "eng", "x"),
        ],
        "doc_id string, stratum string, language_detected string, language_conf double, "
        "cc_language string, text string",
    ).collect()
    record = build_record(
        "CC-MAIN-2026-39", 42, LanguageRules(), {"en_high": 10, "ar_missed": 3}, picked
    )

    assert record["group_sizes"] == {"ar_missed": 3, "en_high": 10}
    assert record["thresholds"] == {"en": 0.65, "fr": 0.65, "ar": 0.65}
    assert [p["doc_id"] for p in record["pages"]] == ["d1", "d2"]  # stable order
    assert record["pages"][1] == {
        "doc_id": "d2",
        "stratum": "ar_missed",
        "language_detected": "arz",
        "language_conf": 0.8,
        "cc_language": "ara",
    }
    assert "secret page text" not in json.dumps(record)
