"""Silver step 2c: text normalization (Story 2.3, T10, DECISIONS S2-06).

Text that looks the same must become the same bytes, or every later hash treats it as
different: boilerplate line hashing (Story 2.2), exact dedup and MinHash (Sprint 4).
So normalize_common runs before any of them (invariant 5), on every document.

Arabic (decision D6, user: option B, DECISIONS S2-09): the published text only loses
what carries no information (tatweel, in normalize_common). The lossy folds (vowel
marks, hamza forms, teh marbuta, alef maqsura, digits) go into arabic_match_key: a
folded copy computed only when comparing texts (stopwords, dedup hashes), never
published and never stored.
"""

import re
import unicodedata

# Invisible characters that NFKC keeps, but that make identical-looking text differ:
#   U+200B-U+200F  zero-width space, non-joiner, joiner, left-to-right / right-to-left marks
#   U+FEFF         byte-order mark (also used as a zero-width no-break space)
#   U+00AD         soft hyphen: "ex<shy>ample" must match "example"
#   U+2060         word joiner
_INVISIBLE = re.compile("[\u200b-\u200f\ufeff\u00ad\u2060]")

# Windows (\r\n) and old Mac (\r) line endings.
_LINE_ENDINGS = re.compile(r"\r\n?")

# Tatweel stretches a word for looks only (a kashida): removing it loses nothing, so it
# is the one Arabic rule applied to the published text (D6). Only Arabic script uses it.
_TATWEEL = "\N{ARABIC TATWEEL}"


def normalize_common(text: str) -> str:
    """The normalization every document gets, whatever its language.

    Order matters: invisible characters go first, so NFKC can compose what they
    separated; NFKC comes before the whitespace rules, because it turns special
    spaces (no-break, narrow no-break) into ordinary ones. Newlines are always kept:
    line structure drives boilerplate removal and the line-based quality signals.
    """
    text = _INVISIBLE.sub("", text)
    text = text.replace(_TATWEEL, "")  # decoration only: the same word without it (D6)
    # NFKC: canonical equivalents composed (e + combining acute -> é) and compatibility
    # variants folded (ligatures, full-width, no-break spaces, Arabic presentation
    # forms). Lossy on purpose, e.g. m² -> m2 (S2-06).
    text = unicodedata.normalize("NFKC", text)
    text = _LINE_ENDINGS.sub("\n", text)
    # Per line: split() with no argument cuts on every run of whitespace (spaces, tabs,
    # ...) and ignores it at both ends; joining with one space collapses the runs and
    # trims the line in a single C-level pass, so "  Menu \t Home " -> "Menu Home"
    # and it hashes like "Menu Home" in boilerplate detection. A regex doing the same
    # was 34% slower overall: it rewrote every single space between words (S2-06).
    return "\n".join(" ".join(line.split()) for line in text.split("\n"))


# --- The Arabic matching key (D6) ----------------------------------------------------------
# Every rule below loses information, which is why none of them touches the published
# text. Each entry of the table maps a character to its folded form (None = delete).

# Harakat, the short-vowel and related marks: fathatan (U+064B) to sukun (U+0652), plus
# the superscript alef (U+0670). Vocalized text (Quran, poetry, teaching) loses its vowels.
_HARAKAT = [chr(code) for code in range(0x064B, 0x0653)] + ["\N{ARABIC LETTER SUPERSCRIPT ALEF}"]

_ALEF = "\N{ARABIC LETTER ALEF}"
_YEH = "\N{ARABIC LETTER YEH}"

_ARABIC_KEY = str.maketrans(
    {
        **{mark: None for mark in _HARAKAT},
        _TATWEEL: None,  # already gone from normalized text; kept so the key stands alone
        # Alef forms -> bare alef. Standard spelling writes the hamza; web text often omits it.
        "\N{ARABIC LETTER ALEF WITH HAMZA ABOVE}": _ALEF,
        "\N{ARABIC LETTER ALEF WITH HAMZA BELOW}": _ALEF,
        "\N{ARABIC LETTER ALEF WITH MADDA ABOVE}": _ALEF,
        "\N{ARABIC LETTER ALEF WASLA}": _ALEF,
        # Hamza carriers (spec, C11) -> the bare carrier letter. The lone hamza stays.
        "\N{ARABIC LETTER WAW WITH HAMZA ABOVE}": "\N{ARABIC LETTER WAW}",
        "\N{ARABIC LETTER YEH WITH HAMZA ABOVE}": _YEH,
        # Policy choices (S2-09), both lossy, some corpora fold the other way round:
        "\N{ARABIC LETTER TEH MARBUTA}": "\N{ARABIC LETTER HEH}",  # the common web spelling
        "\N{ARABIC LETTER ALEF MAKSURA}": _YEH,  # merges "ala" (on) with "Ali" (the name)
        # Eastern Arabic (U+0660-U+0669) and Persian (U+06F0-U+06F9) digits -> 0-9.
        **{chr(0x0660 + digit): str(digit) for digit in range(10)},
        **{chr(0x06F0 + digit): str(digit) for digit in range(10)},
    }
)


def arabic_key_translation() -> tuple[str, str]:
    """The same key as (matching, replacement) strings for Spark's built-in translate().

    Spark's translate(text, matching, replacement) maps the i-th matching character to
    the i-th replacement character and deletes matching characters that have none, so
    the mapped characters come first and the deleted ones (marks, tatweel) last. One
    table, two engines: a test checks Spark's result equals arabic_match_key's.
    """
    mapped = [(chr(code), target) for code, target in _ARABIC_KEY.items() if target is not None]
    deleted = [chr(code) for code, target in _ARABIC_KEY.items() if target is None]
    matching = "".join(source for source, _ in mapped) + "".join(deleted)
    return matching, "".join(str(target) for _, target in mapped)


def arabic_match_key(text: str) -> str:
    """A folded copy of normalized text, for comparing only: never published (D6).

    Two spellings of one Arabic word get the same key (e.g. with or without vowel marks
    or hamza). Only Arabic-script characters change, so it is safe on any document.
    Expects normalize_common's output; idempotent.
    """
    return text.translate(_ARABIC_KEY)
