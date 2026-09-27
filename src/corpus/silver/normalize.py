"""Silver step 2c: text normalization (Story 2.3, T10, DECISIONS S2-06).

Text that looks the same must become the same bytes, or every later hash treats it as
different: boilerplate line hashing (Story 2.2), exact dedup and MinHash (Sprint 4).
So normalize_common runs before any of them (invariant 5), on every document.
The Arabic-only rules (normalize_arabic, decision D6) come later in Story 2.3.
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


def normalize_common(text: str) -> str:
    """The normalization every document gets, whatever its language.

    Order matters: invisible characters go first, so NFKC can compose what they
    separated; NFKC comes before the whitespace rules, because it turns special
    spaces (no-break, narrow no-break) into ordinary ones. Newlines are always kept:
    line structure drives boilerplate removal and the line-based quality signals.
    """
    text = _INVISIBLE.sub("", text)
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
