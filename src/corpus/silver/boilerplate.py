"""Silver step 2b: boilerplate removal (Story 2.2, DECISIONS S2-07).

Two kinds of rules, applied in this order (the order changes the cost, not the result):
1. Per-line rules, in pure Python: they judge one line on its own, so they run inside
   the parse step, before any shuffle.
2. The domain rule, in Spark: a line repeated across many documents of the same domain
   is a template line (menu, footer). Pass 1 finds those lines (a shuffle); pass 2
   removes them from each document (a broadcast, no shuffle).
"""

import unicodedata
from dataclasses import dataclass
from typing import NamedTuple

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

# A line "ends a sentence" if its last character, after closing quotes and brackets,
# is one of these. Includes the Arabic question mark and semicolon. French spacing
# ("Bonjour !") needs nothing special: normalize_common made that space ordinary.
_SENTENCE_END = frozenset(".!?:;\N{ARABIC QUESTION MARK}\N{ARABIC SEMICOLON}")
_CLOSERS = (
    "\"')]}"
    "\N{RIGHT DOUBLE QUOTATION MARK}\N{RIGHT SINGLE QUOTATION MARK}"
    "\N{RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK}"
)


@dataclass(frozen=True)
class LineRules:
    """Per-line thresholds. Source for all three: the implementation plan, task 2.2.4
    (adapted from C4 / Gopher-style line filters); to revisit after the 50-document read."""

    min_words: int = 4  # rule A: fewer words -> removed
    min_words_without_end: int = 10  # rule B: no sentence end and fewer words -> removed
    max_non_letter_share: float = 0.5  # rule C: more non-letters than this -> removed


@dataclass(frozen=True)
class DomainRules:
    """The domain rule's thresholds. Source: the implementation plan, tasks 2.2.2-2.2.3."""

    min_share: float = 0.30  # boilerplate if in MORE than 30% of the domain's documents...
    min_documents: int = 5  # ...and in at least 5 of them (a floor for small domains)
    max_entries: int = 10_000  # broadcast cap for pass 2: keeps the broadcast small


class LineCleaning(NamedTuple):
    text: str  # the kept lines, joined with \n
    few_words: int  # lines removed by rule A
    no_sentence_end: int  # lines removed by rule B (and not A)
    mostly_non_letters: int  # lines removed by rule C (and not A or B)

    @property
    def lines_removed(self) -> int:
        return self.few_words + self.no_sentence_end + self.mostly_non_letters


def has_few_words(line: str, rules: LineRules) -> bool:
    """Rule A: e.g. "Home", "Share this", "Read more >>"."""
    return len(line.split()) < rules.min_words


def lacks_sentence_end(line: str, rules: LineRules) -> bool:
    """Rule B: a short line that does not end like a sentence: menus, tags, headings."""
    end = line.rstrip(_CLOSERS)
    ends_sentence = bool(end) and end[-1] in _SENTENCE_END
    return not ends_sentence and len(line.split()) < rules.min_words_without_end


def is_mostly_non_letters(line: str, rules: LineRules) -> bool:
    """Rule C: prices, dates, counters, code: more than half of the characters are not
    letters. Spaces are ignored, and so are combining marks (Arabic harakat, accents):
    fully vocalized Arabic is 44-57% marks (measured, S2-07), so counting them as
    non-letters would remove Quranic verses and poetry as if they were price lists."""
    letters = others = 0
    for char in line:
        if char.isalpha():
            letters += 1
        elif not char.isspace() and not unicodedata.category(char).startswith("M"):
            others += 1
    counted = letters + others
    return counted > 0 and others / counted > rules.max_non_letter_share


def clean_lines(text: str, rules: LineRules) -> LineCleaning:
    """Apply rules A, B, C to every line of a normalized text.

    Blank lines are dropped without being counted: there is nothing to judge. Each
    removed line is credited to the first rule that catches it, so the three counts
    add up to the total.
    """
    kept: list[str] = []
    few = no_end = non_letters = 0
    for line in text.split("\n"):
        if not line:
            continue
        if has_few_words(line, rules):
            few += 1
        elif lacks_sentence_end(line, rules):
            no_end += 1
        elif is_mostly_non_letters(line, rules):
            non_letters += 1
        else:
            kept.append(line)
    return LineCleaning("\n".join(kept), few, no_end, non_letters)


# --- The domain rule, in Spark (built-in functions only, no Python) ----------------------


def line_hashes(docs: DataFrame) -> DataFrame:
    """One row (domain, line_hash) per distinct non-empty line of each document.

    array_distinct: a line repeated ten times in one page counts once. xxhash64 turns the
    line into an 8-byte number, so the shuffle moves numbers instead of whole lines; it is
    deterministic across machines, unlike Python's hash() (CLAUDE.md section 8).
    """
    lines = F.explode(F.array_distinct(F.split("text", "\n"))).alias("line")
    return (
        docs.select("domain", lines)
        .where(F.col("line") != "")
        .select("domain", F.xxhash64("line").alias("line_hash"))
    )


def domain_boilerplate(docs: DataFrame, rules: DomainRules) -> DataFrame:
    """Pass 1: (domain, line_hash, documents, domain_documents) of every boilerplate line.

    `docs` holds document rows (domain, text), after the per-line rules. The groupBy on
    (domain, line_hash) is the job's first shuffle: every row of one pair must meet on one
    executor. Spark pre-counts on each executor first (partial aggregation), so what
    crosses the network is one row per distinct pair per partition, not one per line.
    Not capped: pass 2 applies max_entries, and the caller counts what the cap cuts off.
    """
    domain_documents = docs.groupBy("domain").agg(F.count("*").alias("domain_documents"))
    line_documents = (
        line_hashes(docs).groupBy("domain", "line_hash").agg(F.count("*").alias("documents"))
    )
    return line_documents.join(domain_documents, "domain").where(
        (F.col("documents") >= rules.min_documents)
        & (F.col("documents") > rules.min_share * F.col("domain_documents"))
    )


def cap_entries(boilerplate: DataFrame, rules: DomainRules) -> DataFrame:
    """The max_entries most frequent boilerplate lines. Ties are broken by domain, then
    hash, so the cap keeps the same entries on every run (determinism, CLAUDE.md section 8)."""
    return boilerplate.orderBy(F.desc("documents"), "domain", "line_hash").limit(rules.max_entries)


def remove_domain_boilerplate(
    docs: DataFrame, boilerplate: DataFrame, rules: DomainRules
) -> DataFrame:
    """Pass 2: drop each document's boilerplate lines. Adds `lines_removed_domain`.

    The capped entries become one small row per domain, (domain, [line hashes]), which
    is broadcast: every executor gets a full copy, so the documents never move (no
    second shuffle). Each document then filters its own lines with Spark's built-in
    `filter`, which keeps the lines in their original order. Documents of domains with
    no boilerplate are returned untouched.
    """
    per_domain = (
        cap_entries(boilerplate, rules)
        .groupBy("domain")
        .agg(F.collect_set("line_hash").alias("boilerplate_hashes"))
    )
    hashes = F.col("boilerplate_hashes")
    lines = F.split("text", "\n")
    kept = F.filter(lines, lambda line: ~F.array_contains(hashes, F.xxhash64(line)))
    return (
        docs.join(F.broadcast(per_domain), "domain", "left")
        .withColumn(
            "lines_removed_domain",
            F.when(hashes.isNull(), F.lit(0)).otherwise(F.size(lines) - F.size(kept)),
        )
        .withColumn(
            "text", F.when(hashes.isNull(), F.col("text")).otherwise(F.array_join(kept, "\n"))
        )
        .drop("boilerplate_hashes")
    )
