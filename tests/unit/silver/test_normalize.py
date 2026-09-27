"""corpus.silver.normalize.normalize_common: one test per rule (task 2.3.3, S2-06).

Invisible characters are written as \\u escapes so a reader can see them.
"""

from pathlib import Path

import pytest

from corpus.silver.normalize import normalize_common
from corpus.silver.parse import Document, parse_wet

SAMPLE = Path(__file__).parents[2] / "fixtures" / "sample.wet.gz"


# Rule 1: invisible characters are removed ------------------------------------------


@pytest.mark.parametrize(
    "invisible",
    [
        "\u200b",  # zero-width space
        "\u200c",  # zero-width non-joiner
        "\u200d",  # zero-width joiner
        "\u200e",  # left-to-right mark
        "\u200f",  # right-to-left mark
        "\ufeff",  # byte-order mark
        "\u00ad",  # soft hyphen
        "\u2060",  # word joiner
    ],
)
def test_invisible_characters_are_removed(invisible: str) -> None:
    assert normalize_common(f"ex{invisible}ample") == "example"


def test_an_invisible_character_no_longer_blocks_composition() -> None:
    # e, zero-width space, combining acute: once the invisible one is gone, NFKC
    # composes e + acute into the single character é.
    assert normalize_common("cafe\u200b\u0301") == "caf\u00e9"


# Rule 2: NFKC ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("\ufb01n", "fin"),  # ligature fi -> two letters
        ("\uff21\uff22\uff23\uff11\uff12\uff13", "ABC123"),  # full-width
        ("a\u00a0b", "a b"),  # no-break space (HTML &nbsp;) -> space
        ("Bonjour\u202f!", "Bonjour !"),  # French narrow no-break space -> space
        ("wait\u2026", "wait..."),  # ellipsis character -> three dots (S3 signal)
        ("\ufefb", "\u0644\u0627"),  # Arabic presentation form lam-alef -> lam + alef
        ("\ufe91", "\u0628"),  # Arabic presentation form of beh -> beh
    ],
)
def test_nfkc_folds_look_alike_variants(before: str, after: str) -> None:
    assert normalize_common(before) == after


def test_visually_identical_strings_become_byte_identical() -> None:
    """The plan's headline check: same look, different bytes, same bytes afterwards."""
    composed = "caf\u00e9"  # é as one code point
    decomposed = "cafe\u0301"  # e + combining acute accent
    assert composed != decomposed  # they look the same but differ in bytes
    assert normalize_common(composed).encode() == normalize_common(decomposed).encode()


def test_nfkc_losses_are_accepted_and_documented() -> None:
    """Known, accepted loss of the compatibility fold (S2-06)."""
    assert normalize_common("5 m\u00b2") == "5 m2"


# Rule 3: line endings -------------------------------------------------------------


def test_windows_and_old_mac_line_endings_become_newlines() -> None:
    assert normalize_common("a\r\nb\rc\nd") == "a\nb\nc\nd"


# Rule 4: horizontal whitespace ------------------------------------------------------


def test_runs_of_spaces_and_tabs_become_one_space() -> None:
    assert normalize_common("a \t  b\t\tc") == "a b c"


def test_each_line_is_trimmed() -> None:
    assert normalize_common("  Menu  \n\tHome\t") == "Menu\nHome"


def test_newlines_are_kept_including_blank_lines() -> None:
    assert normalize_common("para one\n\n\npara two\n") == "para one\n\n\npara two\n"


def test_boilerplate_lines_that_differ_only_invisibly_become_equal() -> None:
    """Why this runs before line hashing (T10): the same menu line, three encodings."""
    variants = ["Home | Contact", "Home\u00a0|\u00a0Contact ", "\tHome | Con\u00adtact"]
    assert {normalize_common(v) for v in variants} == {"Home | Contact"}


# What it must NOT do --------------------------------------------------------------


def test_arabic_letters_marks_and_digits_are_left_to_normalize_arabic() -> None:
    """Harakat, tatweel, alef forms and Eastern digits are real characters, not variants:
    NFKC keeps them. Whether to fold them is decision D6 (normalize_arabic)."""
    arabic = "\u0643\u064e\u062a\u064e\u0628\u064e \u0643\u0640\u062a\u0628 \u0623 \u0663"
    assert normalize_common(arabic) == arabic


def test_normalizing_twice_changes_nothing_on_real_documents() -> None:
    """Idempotent: safe to rerun, and a normalized text is a fixed point."""
    with SAMPLE.open("rb") as stream:
        texts = [d.text for d in parse_wet(stream) if isinstance(d, Document)]
    assert len(texts) == 3
    for text in texts:
        once = normalize_common(text)
        assert normalize_common(once) == once
