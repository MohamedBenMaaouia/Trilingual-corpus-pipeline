"""corpus.silver.boilerplate pass 1: the domain rule (tasks 2.2.1-2.2.2, S2-07)."""

import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from corpus.silver.boilerplate import (
    DomainRules,
    domain_boilerplate,
    line_hashes,
    remove_domain_boilerplate,
)

RULES = DomainRules()
FOOTER = "Subscribe to our weekly newsletter today."


def docs(spark: SparkSession, rows: list[tuple[str, str]]) -> DataFrame:
    return spark.createDataFrame(rows, "domain string, text string")


def hash_of(spark: SparkSession, line: str) -> int:
    """The line's xxhash64, computed by Spark itself."""
    row = spark.range(1).select(F.xxhash64(F.lit(line)).alias("h")).first()
    assert row is not None
    return int(row.h)


def found(frame: DataFrame) -> set[tuple[str, int, int, int]]:
    return {
        (r.domain, r.line_hash, r.documents, r.domain_documents)
        for r in frame.select("domain", "line_hash", "documents", "domain_documents").collect()
    }


def test_a_line_repeated_in_one_document_counts_once(spark: SparkSession) -> None:
    rows = line_hashes(docs(spark, [("a.com", f"{FOOTER}\n{FOOTER}\n{FOOTER}\nOther line.")]))
    assert rows.count() == 2  # FOOTER once, "Other line." once


def test_empty_documents_give_no_lines(spark: SparkSession) -> None:
    assert line_hashes(docs(spark, [("a.com", ""), ("a.com", "\n\n")])).count() == 0


def test_a_footer_in_every_document_of_a_domain_is_boilerplate(spark: SparkSession) -> None:
    rows = [("a.com", f"Article number {i} is about something.\n{FOOTER}") for i in range(6)]
    assert found(domain_boilerplate(docs(spark, rows), RULES)) == {
        ("a.com", hash_of(spark, FOOTER), 6, 6)
    }


def test_small_domains_are_protected_by_the_floor(spark: SparkSession) -> None:
    rows = [("b.com", f"Article {i} text here.\n{FOOTER}") for i in range(4)]  # 4 < 5
    assert found(domain_boilerplate(docs(spark, rows), RULES)) == set()


def test_the_share_must_be_more_than_30_percent_not_equal(spark: SparkSession) -> None:
    thirty = "Seen in exactly six of twenty documents."  # 6/20 = 30%: not boilerplate
    thirty_five = "Seen in exactly seven of twenty documents."  # 7/20 = 35%: boilerplate
    rows = []
    for i in range(20):
        lines = [f"Unique article line number {i}."]
        if i < 6:
            lines.append(thirty)
        if i < 7:
            lines.append(thirty_five)
        rows.append(("c.com", "\n".join(lines)))
    assert found(domain_boilerplate(docs(spark, rows), RULES)) == {
        ("c.com", hash_of(spark, thirty_five), 7, 20)
    }


def test_emptied_documents_still_count_in_the_domain_total(spark: SparkSession) -> None:
    """The denominator is every document of the domain, including those the per-line
    rules emptied: 5 of 20 is 25%, below the threshold."""
    rows = [("d.com", FOOTER) for _ in range(5)] + [("d.com", "") for _ in range(15)]
    assert found(domain_boilerplate(docs(spark, rows), RULES)) == set()


def test_domains_are_judged_separately(spark: SparkSession) -> None:
    """The same footer on two sites is boilerplate only where it is frequent."""
    rows = [("a.com", FOOTER) for _ in range(5)] + [
        ("e.com", f"Story {i} is long enough to stay.") for i in range(10)
    ]
    rows.append(("e.com", FOOTER))  # 1 of 11 on e.com
    assert found(domain_boilerplate(docs(spark, rows), RULES)) == {
        ("a.com", hash_of(spark, FOOTER), 5, 5)
    }


# Pass 2: removal --------------------------------------------------------------------


def clean(frame: DataFrame, rules: DomainRules = RULES) -> DataFrame:
    """Pass 1 then pass 2, as the job runs them."""
    return remove_domain_boilerplate(frame, domain_boilerplate(frame, rules), rules)


def test_pass_2_removes_the_footer_and_keeps_line_order(spark: SparkSession) -> None:
    rows = [
        ("a.com", f"Line one of story {i}.\n{FOOTER}\nLine two of story {i}.") for i in range(5)
    ]
    result = clean(docs(spark, rows)).orderBy("text").collect()
    assert [r.text for r in result] == [
        f"Line one of story {i}.\nLine two of story {i}." for i in range(5)
    ]
    assert all(r.lines_removed_domain == 1 for r in result)


def test_documents_of_other_domains_are_untouched(spark: SparkSession) -> None:
    story = "A story that appears once.\nWith a second sentence."
    rows = [("a.com", FOOTER) for _ in range(5)] + [("e.com", story)]
    other = clean(docs(spark, rows)).where(F.col("domain") == "e.com").first()
    assert other is not None
    assert other.text == story  # byte-for-byte: never split and rejoined
    assert other.lines_removed_domain == 0


def test_a_document_of_only_boilerplate_becomes_empty(spark: SparkSession) -> None:
    rows = [("a.com", FOOTER) for _ in range(5)]
    assert {r.text for r in clean(docs(spark, rows)).collect()} == {""}


def test_the_cap_keeps_the_most_frequent_lines(spark: SparkSession) -> None:
    """With room for one entry, the line in 6 documents wins over the one in 5."""
    rare = "In five of the six documents."
    rows = [("a.com", f"{FOOTER}\n{rare}" if i < 5 else FOOTER) for i in range(6)]
    result = clean(docs(spark, rows), DomainRules(max_entries=1)).collect()
    assert sorted(r.text for r in result) == ["", rare, rare, rare, rare, rare]


def test_pass_2_is_a_broadcast_join(
    spark: SparkSession, capsys: pytest.CaptureFixture[str]
) -> None:
    """The evidence that documents do not move: the plan joins by broadcast."""
    rows = [("a.com", FOOTER) for _ in range(5)]
    clean(docs(spark, rows)).explain()
    assert "BroadcastHashJoin" in capsys.readouterr().out
