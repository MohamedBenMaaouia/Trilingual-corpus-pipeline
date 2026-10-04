"""Silver step 2e: quality signals, reject reasons, score and tier (Story 3.2, DECISIONS S3-08).

The seven signals of the plan, computed with Spark built-ins (no Python per row). The
rules are Gopher's (Rae et al. 2021, appendix A.1.1), kept as they are for the
structure-based checks, and calibrated per language for word length (S3-08). Every
reject reason is kept (an array): a page can fail several checks.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cache
from importlib import resources

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from corpus.silver.normalize import arabic_key_translation, arabic_match_key

# Reason codes (controlled vocabulary, pipeline.md section 10), one per check.
TOO_FEW_WORDS = "too_few_words"
TOO_MANY_WORDS = "too_many_words"
WORD_LENGTH = "word_length_out_of_range"
TOO_MANY_SYMBOLS = "too_many_symbols"
TOO_FEW_STOPWORDS = "too_few_stopwords"
REPEATED_LINES = "too_many_repeated_lines"
ELLIPSIS_LINES = "too_many_ellipsis_lines"

HIGH, MEDIUM, REJECTED = "high", "medium", "rejected"


@dataclass(frozen=True)
class Thresholds:
    """One language's limits. Defaults: Gopher's values (appendix A.1.1)."""

    min_words: int = 50  # Gopher: 50 to 100,000 words
    max_words: int = 100_000  # Gopher
    min_mean_word_length: float = 3.0  # Gopher: mean word length 3 to 10 (English)
    max_mean_word_length: float = 10.0  # Gopher (English)
    max_symbol_ratio: float = 0.1  # Gopher: "#" or "..." to word ratio at most 0.1
    min_stopwords: int = 2  # Gopher: at least 2 of the language's stopword list
    max_repeated_line_ratio: float = 0.3  # Gopher: duplicate-line fraction at most 0.3
    max_ellipsis_line_ratio: float = 0.3  # Gopher: lines ending in "..." at most 30%


def _default_thresholds() -> dict[str, Thresholds]:
    # Mean word length for fr and ar: Gopher's English bounds (3, 10) scaled by the
    # language's median mean word length over English's, measured on the dev run
    # (DECISIONS S3-08): en 5.2202, fr 5.3304 (x1.0211), ar 5.0662 (x0.9705). Everything
    # else is Gopher's and language-neutral.
    return {
        "en": Thresholds(),
        "fr": Thresholds(min_mean_word_length=3.06, max_mean_word_length=10.21),
        "ar": Thresholds(min_mean_word_length=2.91, max_mean_word_length=9.70),
    }


@dataclass(frozen=True)
class QualityRules:
    per_language: Mapping[str, Thresholds] = field(default_factory=_default_thresholds)
    # Checks that do not depend on a language apply to every page with these limits;
    # word length and stopwords only apply to en, fr and ar.
    default: Thresholds = field(default_factory=Thresholds)
    # The stopword ratio counted as "fully normal" by the score, per language: the
    # median over confident pages of at least 50 words on the dev run (DECISIONS S3-08).
    typical_stopword_ratio: Mapping[str, float] = field(
        default_factory=lambda: {"en": 0.1329, "fr": 0.1603, "ar": 0.0849}
    )
    # Words at which the length part of the score reaches 1 (provisional, S3-08).
    full_score_words: int = 1_000
    # quality_score at or above this is "high", below is "medium" (provisional, D10).
    high_score: float = 0.8


@cache
def stopwords(language: str) -> tuple[str, ...]:
    """The language's stopword list, folded like the text is when comparing: lowercase
    and arabic_match_key (trap S3: else "على" never matches the web's "علي")."""
    path = resources.files("corpus") / "resources" / "stopwords" / f"{language}.txt"
    lines = path.read_text(encoding="utf-8").splitlines()
    words = [arabic_match_key(line.strip().lower()) for line in lines]
    return tuple(w for w in words if w and not w.startswith("#"))


def _ratio(numerator: Column, denominator: Column) -> Column:
    """numerator / denominator as a double, 0.0 when the denominator is 0 (empty page)."""
    return F.when(denominator > 0, numerator / denominator).otherwise(F.lit(0.0))


def quality_signals(docs: DataFrame, rules: QualityRules) -> DataFrame:
    """Add the 7 signals (plan 3.2.1) and the helper column stopword_count.

    Needs `text` (normalized, boilerplate removed, PII redacted) and `language`.
    Words are the pieces between whitespace; lines the non-empty pieces between "\\n".
    """
    words = F.filter(F.split("text", r"\s+"), lambda w: w != "")
    lines = F.filter(F.split("text", "\n"), lambda line: line != "")
    word_count = F.size(words)
    line_count = F.size(lines)
    # "#" and "..." counted separately; Gopher rejects when either ratio is above 0.1,
    # so the signal is the larger one. NFKC already turned "…" into "..." (S2-06).
    hashes = F.size(F.split("text", "#")) - 1
    ellipses = (F.length("text") - F.length(F.regexp_replace("text", r"\.\.\.", ""))) / 3
    # A word as compared with the list: lowercase, punctuation stripped at both ends
    # ("the," -> "the"), Arabic folded with the same key as the list (translate() is
    # Spark's built-in; a test checks it equals arabic_match_key).
    matching, replacement = arabic_key_translation()
    folded = F.transform(
        words,
        lambda w: F.translate(
            F.lower(F.regexp_replace(w, r"^\p{P}+|\p{P}+$", "")), matching, replacement
        ),
    )
    lists = F.create_map(
        *[
            item
            for language in rules.per_language
            for item in (F.lit(language), F.array(*[F.lit(w) for w in stopwords(language)]))
        ]
    )
    own_list = lists[F.col("language")]  # null for 'other': no list, no stopwords
    stopword_count = F.when(own_list.isNull(), F.lit(0)).otherwise(
        F.size(F.filter(folded, lambda w: F.array_contains(own_list, w)))
    )
    return (
        docs.withColumn("char_count", F.length("text").cast("int"))
        .withColumn("word_count", word_count.cast("int"))
        .withColumn(
            "mean_word_length",
            _ratio(F.length(F.regexp_replace("text", r"\s", "")), F.col("word_count")),
        )
        .withColumn(
            "symbol_to_word_ratio", _ratio(F.greatest(hashes, ellipses), F.col("word_count"))
        )
        .withColumn("stopword_count", stopword_count.cast("int"))
        .withColumn("stopword_ratio", _ratio(F.col("stopword_count"), F.col("word_count")))
        .withColumn(
            "repeated_line_ratio", _ratio(line_count - F.size(F.array_distinct(lines)), line_count)
        )
        .withColumn(
            "ellipsis_line_ratio",
            _ratio(F.size(F.filter(lines, lambda line: line.endswith("..."))), line_count),
        )
    )


def _limit(rules: QualityRules, name: str) -> Column:
    """The page's own limit for one threshold: its language's, else the default."""
    value: Column = F.lit(getattr(rules.default, name))
    for language, limits in rules.per_language.items():
        value = F.when(F.col("language") == language, F.lit(getattr(limits, name))).otherwise(value)
    return value


def _clip(column: Column) -> Column:
    return F.least(F.greatest(column, F.lit(0.0)), F.lit(1.0))


def classify(docs: DataFrame, rules: QualityRules) -> DataFrame:
    """reject_reasons (array or null), quality_score (0..1) and quality_tier.

    Needs the signals, `language` and `language_reject_reason` (Story 3.1).
    - Reasons: the language reason, then one per failed check. Word length and
      stopwords only judge en, fr and ar (no list or bounds for other languages).
    - Score: the mean of five parts, each 0..1, 1 = clearly fine: length (log scale,
      from min_words to full_score_words), stopwords (ratio vs the language's typical
      ratio), and how far the page is from the repeated-line, ellipsis and symbol
      limits. 0.0 for pages not in a target language (T12).
    - Tier: rejected if any reason, else high at or above high_score, else medium.
    """
    target = F.col("language").isin(*rules.per_language)
    words, stops = F.col("word_count"), F.col("stopword_count")
    checks = [
        (words < _limit(rules, "min_words"), TOO_FEW_WORDS),
        (words > _limit(rules, "max_words"), TOO_MANY_WORDS),
        (
            target
            & (
                (F.col("mean_word_length") < _limit(rules, "min_mean_word_length"))
                | (F.col("mean_word_length") > _limit(rules, "max_mean_word_length"))
            ),
            WORD_LENGTH,
        ),
        (F.col("symbol_to_word_ratio") > _limit(rules, "max_symbol_ratio"), TOO_MANY_SYMBOLS),
        (target & (stops < _limit(rules, "min_stopwords")), TOO_FEW_STOPWORDS),
        (
            F.col("repeated_line_ratio") > _limit(rules, "max_repeated_line_ratio"),
            REPEATED_LINES,
        ),
        (
            F.col("ellipsis_line_ratio") > _limit(rules, "max_ellipsis_line_ratio"),
            ELLIPSIS_LINES,
        ),
    ]
    reasons = F.filter(
        F.array(
            F.col("language_reject_reason"),
            *[F.when(failed, F.lit(code)) for failed, code in checks],
        ),
        lambda reason: reason.isNotNull(),
    )

    typical = F.lit(None).cast("double")
    for language, ratio in rules.typical_stopword_ratio.items():
        typical = F.when(F.col("language") == language, F.lit(ratio)).otherwise(typical)
    low, high = F.log10(_limit(rules, "min_words")), F.log10(F.lit(rules.full_score_words))
    parts = [
        _clip((F.log10(F.greatest(words, F.lit(1))) - low) / (high - low)),
        _clip(F.col("stopword_ratio") / typical),
        _clip(1 - F.col("repeated_line_ratio") / _limit(rules, "max_repeated_line_ratio")),
        _clip(1 - F.col("ellipsis_line_ratio") / _limit(rules, "max_ellipsis_line_ratio")),
        _clip(1 - F.col("symbol_to_word_ratio") / _limit(rules, "max_symbol_ratio")),
    ]
    score = F.when(target & (words > 0), sum(parts[1:], parts[0]) / len(parts)).otherwise(0.0)

    return (
        docs.withColumn("reject_reasons", F.when(F.size(reasons) > 0, reasons))
        .withColumn("quality_score", F.round(score, 4))
        .withColumn(
            "quality_tier",
            F.when(F.col("reject_reasons").isNotNull(), F.lit(REJECTED))
            .when(F.col("quality_score") >= rules.high_score, F.lit(HIGH))
            .otherwise(F.lit(MEDIUM)),
        )
    )
