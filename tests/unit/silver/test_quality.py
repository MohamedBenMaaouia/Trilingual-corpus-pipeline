"""corpus.silver.quality: signals, reasons, score, tier (Story 3.2, DECISIONS S3-08)."""

import pytest
from pyspark.sql import Row, SparkSession
from pyspark.sql import functions as F

from corpus.silver.normalize import arabic_key_translation, arabic_match_key
from corpus.silver.pii import redact_pii
from corpus.silver.quality import (
    ELLIPSIS_LINES,
    REPEATED_LINES,
    TOO_FEW_STOPWORDS,
    TOO_FEW_WORDS,
    TOO_MANY_SYMBOLS,
    WORD_LENGTH,
    QualityRules,
    classify,
    quality_signals,
    stopwords,
)

RULES = QualityRules()
SENTENCE = "the cat and the dog have to be with that friend of mine"  # 13 words, 9 stopwords


def run(spark: SparkSession, rows: list[tuple[str, str, str | None, str]]) -> dict[str, Row]:
    docs = spark.createDataFrame(
        rows, "doc_id string, language string, language_reject_reason string, text string"
    )
    result = classify(quality_signals(docs, RULES), RULES)
    return {r.doc_id: r for r in result.collect()}  # a few test rows


def test_signals_on_a_known_text(spark: SparkSession) -> None:
    text = "\n".join([SENTENCE, SENTENCE, "news # more ..."])
    row = run(spark, [("d", "en", None, text)])["d"]
    assert row.word_count == 30 and row.char_count == len(text)
    assert row.stopword_count == 18 and row.stopword_ratio == pytest.approx(18 / 30)
    assert row.repeated_line_ratio == pytest.approx(1 / 3)  # 3 lines, 1 is a repeat
    assert row.ellipsis_line_ratio == pytest.approx(1 / 3)
    assert row.symbol_to_word_ratio == pytest.approx(1 / 30)  # one "#", one "..."
    letters = len(text.replace(" ", "").replace("\n", ""))
    assert row.mean_word_length == pytest.approx(letters / 30)


def test_punctuation_and_case_do_not_hide_stopwords(spark: SparkSession) -> None:
    row = run(spark, [("d", "en", None, "The, cat. AND: dog!")])["d"]
    assert row.stopword_count == 2


def test_arabic_stopwords_match_through_the_key(spark: SparkSession) -> None:
    # The web writes "على" as "علي" and "إلى" as "الى": both still count (trap S3).
    row = run(spark, [("d", "ar", None, "ذهب علي الى المدرسة في الصباح")])["d"]
    assert row.stopword_count == 3


def test_spark_translate_equals_the_python_key_on_every_arabic_character(
    spark: SparkSession,
) -> None:
    chars = [chr(c) for c in range(0x0600, 0x0700)] + ["a", "é", "5"]
    matching, replacement = arabic_key_translation()
    df = spark.createDataFrame([(c,) for c in chars], "c string")
    folded = df.select("c", F.translate("c", matching, replacement).alias("k")).collect()
    assert {r.c: r.k for r in folded} == {c: arabic_match_key(c) for c in chars}


def test_stopword_lists_are_folded_like_the_text() -> None:
    assert stopwords("en") == ("the", "be", "to", "of", "and", "that", "have", "with")
    assert all(w == arabic_match_key(w) for w in stopwords("ar"))


def test_a_good_page_is_kept_with_no_reasons(spark: SparkSession) -> None:
    text = "\n".join(f"{SENTENCE} line {i}." for i in range(5))  # 70 words
    row = run(spark, [("d", "en", None, text)])["d"]
    assert row.reject_reasons is None
    assert row.quality_tier in ("high", "medium") and 0 < row.quality_score <= 1


def test_each_check_gives_its_reason(spark: SparkSession) -> None:
    long_words = " ".join(["incomprehensibilities"] * 60)  # 21 letters per word
    symbols = " ".join(["#tag the and word"] * 20)  # 80 words, 20 "#": ratio 0.25
    repeated = "\n".join([SENTENCE] * 5)  # 65 words, 4 of 5 lines repeat
    ellipsis = "\n".join(f"{SENTENCE} {i} ..." for i in range(5))
    rows = run(
        spark,
        [
            ("short", "en", None, SENTENCE),
            ("long-words", "en", None, long_words),
            ("symbols", "en", None, symbols),
            ("no-stopwords", "en", None, " ".join(["word"] * 60)),
            ("repeated", "en", None, repeated),
            ("ellipsis", "en", None, ellipsis),
        ],
    )
    assert TOO_FEW_WORDS in rows["short"].reject_reasons
    assert WORD_LENGTH in rows["long-words"].reject_reasons
    assert TOO_MANY_SYMBOLS in rows["symbols"].reject_reasons
    assert rows["no-stopwords"].reject_reasons == [TOO_FEW_STOPWORDS]
    assert REPEATED_LINES in rows["repeated"].reject_reasons
    assert ELLIPSIS_LINES in rows["ellipsis"].reject_reasons
    assert all(r.quality_tier == "rejected" for r in rows.values())


def test_language_reason_comes_first_and_other_languages_score_zero(spark: SparkSession) -> None:
    text = "\n".join(f"{SENTENCE} line {i}." for i in range(5))
    rows = run(
        spark, [("de", "other", "language_not_targeted", text), ("empty", "other", None, "")]
    )
    assert rows["de"].reject_reasons == ["language_not_targeted"]
    assert rows["de"].quality_score == 0.0  # T12
    # An emptied page: no language reason, rejected by length, every signal 0.
    assert rows["empty"].reject_reasons == [TOO_FEW_WORDS]
    assert (rows["empty"].word_count, rows["empty"].mean_word_length) == (0, 0.0)


def test_word_length_bounds_are_per_language(spark: SparkSession) -> None:
    # (60 x 10 + 2 x 3) / 62 = 9.77 letters per word: inside English's 3-10, above
    # Arabic's 2.91-9.70 (S3-08).
    text = " ".join(["abcdefghij"] * 60 + ["the", "and"])
    rows = run(spark, [("en", "en", None, text), ("ar", "ar", None, text)])
    assert WORD_LENGTH not in (rows["en"].reject_reasons or [])
    assert WORD_LENGTH in rows["ar"].reject_reasons


def test_redaction_runs_once_in_python_and_keeps_columns(spark: SparkSession) -> None:
    docs = spark.createDataFrame(
        [("d1", "mail me: a.b@example.fr"), ("d2", "nothing here")], "doc_id string, text string"
    )
    result = redact_pii(docs)
    rows = {r.doc_id: r for r in result.collect()}
    assert (rows["d1"].text, rows["d1"].pii_redacted, rows["d1"].pii_types) == (
        "mail me: [EMAIL]",
        True,
        ["email"],
    )
    assert (rows["d2"].pii_redacted, rows["d2"].pii_types) == (False, None)
    assert result.columns == ["doc_id", "text", "pii_redacted", "pii_types"]
    plan = result._jdf.queryExecution().executedPlan().toString()
    assert plan.count("ArrowEvalPython") == 1
