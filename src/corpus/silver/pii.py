"""Silver step 2f: personal data redaction (Story 3.3; D9, narrowed by S3-05).

Emails (regex) and phone numbers (found and validated by `phonenumbers`, Google's
libphonenumber in Python), replaced by typed placeholders. A bare digit regex would
redact dates, prices and product codes; validation checks the number really fits a
country's numbering plan. No national ID numbers (cut, S3-05). Applied to every row,
rejects included (invariant 4, T12).
"""

import re
from collections.abc import Sequence
from typing import NamedTuple

import pandas as pd
from phonenumbers import NumberParseException, is_valid_number, parse
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.pandas.functions import pandas_udf
from pyspark.sql.types import BooleanType, StringType, StructField, StructType

EMAIL, PHONE = "email", "phone"  # pii_types values (controlled vocabulary)
PLACEHOLDERS = {EMAIL: "[EMAIL]", PHONE: "[PHONE]"}

# D9: French and Tunisian numbers in national format ("06 12 34 56 78", "71 234 567").
# International numbers ("+33 6 12 34 56 78", E.164) parse whatever the region.
PHONE_REGIONS = ("FR", "TN")

# name@domain.tld: the domain needs a dot and a 2+ letter ending.
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")

# Phone-shaped pieces of text: an optional "+" or "00", then 8 to 15 digits, possibly
# separated by up to 2 spaces, dots, dashes, slashes or brackets, and not glued to a
# longer number or a word. Every number of the covered countries has at least 8 digits
# (Tunisia 8, France 10 nationally). Only these pieces are validated: running the
# library's matcher over whole pages was ~5 ms per page (S3-08).
_PHONE_SHAPE = re.compile(r"(?<![\w+])(?:\+|00)?\d(?:[ .\-/()]{0,2}\d){7,14}(?!\w)")

# Dates are the most common 8-digit pieces of web text ("2026-09-04", "04/09/2026"), and
# "20 26 09 04" is a valid Tunisian mobile number: a date-shaped piece is never a phone.
_DATE_SHAPE = re.compile(r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{4}")


class Span(NamedTuple):
    start: int
    end: int  # exclusive
    kind: str  # EMAIL or PHONE


def is_phone(piece: str) -> bool:
    """A phone-shaped piece that is a real number: valid in France or Tunisia (or, with
    "+" or "00", in its own country) according to libphonenumber's numbering-plan
    metadata (length, prefixes); never a date, never a bare id-like digit run."""
    if _DATE_SHAPE.search(piece):  # also "2025-7-12 08": a date followed by the hour
        return False
    if piece.isdigit() and not piece.startswith("0"):
        # A bare run of digits is mostly an id or a code. Only a French national number
        # (trunk prefix 0, "0612345678") or "00..." counts; Tunisian numbers, which have
        # no prefix, need their usual spacing ("71 234 567") or "+216" (S3-08).
        return False
    for region in PHONE_REGIONS:
        try:
            if is_valid_number(parse(piece, region)):
                return True
        except NumberParseException:
            continue
    return False


def find_pii(text: str) -> list[Span]:
    """Every email and valid phone number in `text`, in order, never overlapping."""
    spans = [Span(m.start(), m.end(), EMAIL) for m in _EMAIL.finditer(text)]
    spans += [
        Span(m.start(), m.end(), PHONE) for m in _PHONE_SHAPE.finditer(text) if is_phone(m.group())
    ]
    # An email can contain digits that look like a phone: keep the first, longest span
    # and drop anything overlapping a span already kept.
    kept: list[Span] = []
    for span in sorted(spans, key=lambda s: (s.start, -s.end)):
        if not kept or span.start >= kept[-1].end:
            kept.append(span)
    return kept


class Redaction(NamedTuple):
    text: str
    types: list[str]  # sorted, distinct: [], ["email"], ["phone"], ["email", "phone"]


def redact(text: str, spans: Sequence[Span] | None = None) -> Redaction:
    """`text` with each span replaced by its placeholder."""
    found = find_pii(text) if spans is None else spans
    parts, position = [], 0
    for span in found:
        parts += [text[position : span.start], PLACEHOLDERS[span.kind]]
        position = span.end
    parts.append(text[position:])
    return Redaction("".join(parts), sorted({span.kind for span in found}))


# --- In Spark ----------------------------------------------------------------------------

# What the Python step returns per document. pii_types travels as "email,phone" and is
# split back in Spark: one plain string column through Arrow, nothing nested. All
# nullable, as Arrow results always are (S3-04); silver_v1's contract is checked later.
_PII_RESULT = StructType(
    [
        StructField("text", StringType(), True),
        StructField("pii_redacted", BooleanType(), True),
        StructField("pii_types", StringType(), True),
    ]
)


def redact_pii(docs: DataFrame) -> DataFrame:
    """Replace `text` with its redacted version; add pii_redacted and pii_types.

    A pandas_udf on `text` (batches, no per-row call); the other columns stay in Spark.
    pii_types is null when nothing was found (the contract's nullable array).
    """

    # PySpark 3.5's type hints list no StructType return for pandas_udf (see
    # silver.language.detect_languages): a struct result is valid at runtime.
    @pandas_udf(_PII_RESULT)  # type: ignore[call-overload, untyped-decorator]
    def batch(texts: pd.Series) -> pd.DataFrame:
        rows = []
        for text in texts:
            result = redact(text or "")
            rows.append((result.text, bool(result.types), ",".join(result.types)))
        return pd.DataFrame(rows, columns=_PII_RESULT.fieldNames()).astype(
            {"text": object, "pii_redacted": bool, "pii_types": object}
        )

    result = docs.withColumn("_pii", batch("text"))
    types = F.col("_pii.pii_types")
    return result.select(
        *[F.col(c) for c in docs.columns if c != "text"],
        F.col("_pii.text").alias("text"),
        F.col("_pii.pii_redacted").alias("pii_redacted"),
        F.when(types == "", F.lit(None)).otherwise(F.split(types, ",")).alias("pii_types"),
    )
