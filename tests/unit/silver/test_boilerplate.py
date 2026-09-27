"""corpus.silver.boilerplate per-line rules A, B, C (task 2.2.4, S2-07).

French and Arabic lines are written as visible text; only invisible characters are
ever escaped (S2-06).
"""

import pytest

from corpus.silver.boilerplate import (
    LineRules,
    clean_lines,
    has_few_words,
    is_mostly_non_letters,
    lacks_sentence_end,
)

RULES = LineRules()


# Rule A: fewer than 4 words ---------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "removed"),
    [
        ("Home", True),
        ("Share this page", True),  # 3 words
        ("Accueil", True),
        ("الصفحة الرئيسية", True),  # "home page"
        ("This line has four", False),  # exactly 4 words: kept by this rule
    ],
)
def test_rule_a_few_words(line: str, removed: bool) -> None:
    assert has_few_words(line, RULES) is removed


# Rule B: no sentence end and fewer than 10 words -------------------------------------


@pytest.mark.parametrize(
    ("line", "removed"),
    [
        ("Latest news from our team", True),  # a menu or heading
        ("The meeting starts at noon.", False),
        ('He said: "we will stop."', False),  # closing quote ignored, then "."
        ("Est-ce que vous venez ce soir ?", False),  # French spacing before "?"
        ("Voir tous les articles du blog", True),
        ("هل ستأتي إلى الاجتماع غدا؟", False),  # Arabic question mark ends a sentence
        ("آخر الأخبار من فريقنا اليوم", True),  # "latest news from our team today"
        # 10 words or more is kept even without punctuation: long running text.
        ("one two three four five six seven eight nine ten", False),
    ],
)
def test_rule_b_no_sentence_end(line: str, removed: bool) -> None:
    assert lacks_sentence_end(line, RULES) is removed


# Rule C: more than half of the characters are not letters ----------------------------


@pytest.mark.parametrize(
    ("line", "removed"),
    [
        ("Price: 12.99 EUR - 45.00 USD", True),
        ("12/09/2026 14:30 | 1,234 views", True),
        ("The committee met on 12 September 2026.", False),
        ("Le comité s'est réuni le 12 septembre.", False),  # accents are letters
        ("كتب الطالب الدرس في المكتبة.", False),
    ],
)
def test_rule_c_mostly_non_letters(line: str, removed: bool) -> None:
    assert is_mostly_non_letters(line, RULES) is removed


def test_rule_c_does_not_count_harakat_as_non_letters() -> None:
    """Vocalized Arabic is 44-57% diacritic marks (measured, S2-07). Quran 1:2 is exactly
    half; with its verse number, a naive count puts it above 50% and removes it like a
    price list."""
    verse = "ٱلْحَمْدُ لِلَّهِ رَبِّ ٱلْعَٰلَمِينَ (2)"
    chars = [c for c in verse if not c.isspace()]
    naive_share = sum(not c.isalpha() for c in chars) / len(chars)
    assert naive_share > 0.5  # counting marks as non-letters would remove the verse
    assert is_mostly_non_letters(verse, RULES) is False  # marks are left out: kept


# clean_lines: the three rules together ----------------------------------------------


def test_clean_lines_keeps_sentences_and_counts_each_removal_once() -> None:
    text = "\n".join(
        [
            "Home",  # A
            "Latest news from our team",  # B
            "Price: 12.99 EUR - 45.00 USD - 3.50 GBP - 1.00 CHF - 9 JPY - 2 AUD",  # C
            "",  # blank: dropped, not counted
            "The committee met on Friday and approved the new budget.",  # kept
            "Le comité s'est réuni vendredi et a voté le budget.",  # kept
        ]
    )
    result = clean_lines(text, RULES)
    assert result.text == (
        "The committee met on Friday and approved the new budget.\n"
        "Le comité s'est réuni vendredi et a voté le budget."
    )
    assert (result.few_words, result.no_sentence_end, result.mostly_non_letters) == (1, 1, 1)
    assert result.lines_removed == 3


def test_a_line_caught_by_several_rules_is_credited_to_the_first() -> None:
    result = clean_lines("12:30", RULES)  # A (1 word) and C (digits): credited to A
    assert (result.few_words, result.no_sentence_end, result.mostly_non_letters) == (1, 0, 0)


def test_a_document_of_only_boilerplate_becomes_empty() -> None:
    result = clean_lines("Home\nAbout us\nContact", RULES)
    assert result.text == ""
    assert result.lines_removed == 3


def test_known_risk_short_poetry_lines_are_removed() -> None:
    """Documents the risk the 50-document read must judge (plan, Sprint 2 risks):
    half-verses of Arabic poetry are short lines without punctuation, so rule B
    removes them. Not a bug in the rule, a trade-off we have to look at."""
    verse = "يا ليل الصب متى غده"  # 5 words, no punctuation
    result = clean_lines(verse, RULES)
    assert result.text == ""
    assert result.no_sentence_end == 1


def test_thresholds_are_config_not_literals() -> None:
    lenient = LineRules(min_words=1, min_words_without_end=1, max_non_letter_share=1.0)
    assert clean_lines("Latest news", lenient).text == "Latest news"
