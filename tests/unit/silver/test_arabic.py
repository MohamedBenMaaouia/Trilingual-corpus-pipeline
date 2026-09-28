"""Arabic normalization, decision D6 (Story 2.3, DECISIONS S2-09): Sprint 2's headline suite.

Published text: only tatweel is removed (normalize_common), nothing else changes.
Matching key (arabic_match_key): every lossy fold, one test per rule, each with
hand-written before/after pairs. Arabic is written as visible text; invisible or
combining characters that must be seen are written as \\N{...}.
"""

import pytest

from corpus.silver.normalize import arabic_match_key, normalize_common


def key(text: str) -> str:
    """How the pipeline builds a key: normalize first, then fold (invariant 5)."""
    return arabic_match_key(normalize_common(text))


# --- Published text: correct spelling is kept -------------------------------------------


def test_tatweel_is_removed_from_the_published_text() -> None:
    assert normalize_common("كـــتاب جـميل") == "كتاب جميل"


@pytest.mark.parametrize(
    "text",
    ["كَتَبَ", "أحمد", "إسلام", "آمن", "مؤمن", "سئل", "مدرسة", "على", "٢٠٢٦"],
)
def test_the_published_text_keeps_correct_spelling(text: str) -> None:
    """Vowel marks, hamza, teh marbuta, alef maqsura and Eastern digits all stay (D6)."""
    assert normalize_common(text) == text


# --- The matching key, one rule at a time ------------------------------------------------


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("كَتَبَ", "كتب"),  # fatha
        ("قُلْ", "قل"),  # damma, sukun
        ("مُدَرِّسٌ", "مدرس"),  # shadda, dammatan (tanwin)
        ("هٰذا", "هذا"),  # superscript alef
    ],
)
def test_key_removes_harakat(before: str, after: str) -> None:
    assert key(before) == after


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("أحمد", "احمد"),  # alef with hamza above
        ("إسلام", "اسلام"),  # alef with hamza below
        ("آمن", "امن"),  # alef with madda
        ("ٱلحمد", "الحمد"),  # alef wasla
    ],
)
def test_key_folds_alef_forms_to_bare_alef(before: str, after: str) -> None:
    assert key(before) == after


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("مؤمن", "مومن"),  # waw with hamza -> waw
        ("سئل", "سيل"),  # yeh with hamza -> yeh
        ("سماء", "سماء"),  # the lone hamza is not a carrier: kept
    ],
)
def test_key_folds_hamza_carriers(before: str, after: str) -> None:
    assert key(before) == after


def test_key_folds_teh_marbuta_to_heh() -> None:
    assert key("مدرسة") == "مدرسه"  # the common web spelling


@pytest.mark.parametrize(("before", "after"), [("على", "علي"), ("مستشفى", "مستشفي")])
def test_key_folds_alef_maqsura_to_yeh(before: str, after: str) -> None:
    assert key(before) == after


def test_the_accepted_collision_on_and_ali_share_a_key() -> None:
    """Stated openly (S2-09): "ala" (on) and "Ali" (the name) become one key. Harmless
    for dedup (whole documents) and stopwords; the reason it never touches published text."""
    assert key("على") == key("علي")


@pytest.mark.parametrize(("before", "after"), [("٢٠٢٦", "2026"), ("۲۰۲۶", "2026")])
def test_key_folds_eastern_and_persian_digits(before: str, after: str) -> None:
    assert key(before) == after


# --- What the key is for ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("careful", "casual"),
    [
        ("أَحْمَدُ فِي المَدْرَسَةِ", "احمد في المدرسه"),  # vowels, hamza, teh marbuta
        ("ذهب إلى المستشفى عام ٢٠٢٦", "ذهب الي المستشفي عام 2026"),  # hamza, maqsura, digits
    ],
)
def test_two_spellings_of_the_same_sentence_share_a_key(careful: str, casual: str) -> None:
    assert careful != casual
    assert key(careful) == key(casual)


def test_visually_identical_arabic_becomes_byte_identical() -> None:
    """Task 2.3.3: same look, different bytes, same bytes after normalization."""
    pairs = [
        ("\N{ARABIC LETTER ALEF}\N{ARABIC HAMZA ABOVE}حمد", "أحمد"),  # decomposed hamza
        ("\N{ARABIC LIGATURE LAM WITH ALEF ISOLATED FORM}", "لا"),  # presentation form
        ("كـتاب", "كتاب"),  # tatweel
    ]
    for looks_like, standard in pairs:
        assert looks_like.encode() != standard.encode()
        assert normalize_common(looks_like).encode() == normalize_common(standard).encode()


# --- Safety -----------------------------------------------------------------------------


def test_the_key_changes_nothing_outside_the_arabic_block() -> None:
    """Safe on any document: sweep every basic Unicode character (surrogates excluded)."""
    changed = [
        code
        for code in range(0x10000)
        if not 0xD800 <= code <= 0xDFFF and arabic_match_key(chr(code)) != chr(code)
    ]
    assert changed  # the sweep really exercises the table
    assert all(0x0600 <= code <= 0x06FF for code in changed)


def test_english_and_french_are_untouched_by_the_key() -> None:
    text = "Le café coûte 3 euros. The meeting is at 10."
    assert key(text) == normalize_common(text)


@pytest.mark.parametrize("text", ["أَحْمَدُ فِي المَدْرَسَةِ", "على ٢٠٢٦", "كتاب"])
def test_the_key_is_idempotent(text: str) -> None:
    assert arabic_match_key(key(text)) == key(text)
