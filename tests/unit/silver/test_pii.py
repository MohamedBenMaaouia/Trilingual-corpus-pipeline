"""corpus.silver.pii: email and phone redaction (Story 3.3; D9 narrowed by S3-05, S3-08)."""

import pytest

from corpus.silver.pii import EMAIL, PHONE, Redaction, find_pii, is_phone, redact


@pytest.mark.parametrize(
    "piece",
    [
        "06 12 34 56 78",  # French mobile, national format
        "01.42.68.53.00",  # French landline, dots
        "0612345678",  # French, no spaces: the trunk prefix 0 makes it a phone
        "+33 6 12 34 56 78",  # international (E.164 written with spaces)
        "0033 6 12 34 56 78",  # international with 00
        "71 234 567",  # Tunisian landline, usual grouping
        "+216 98 765 432",  # Tunisian mobile, international
        "+1 202 555 0143",  # any country, with +
    ],
)
def test_real_phone_numbers_are_phones(piece: str) -> None:
    assert is_phone(piece)


@pytest.mark.parametrize(
    "piece",
    [
        "2026-09-04",  # a date (8 digits: "20 26 09 04" is a valid Tunisian mobile)
        "04/09/2026",
        "2025-7-12 08",  # a date and the hour
        "98765432",  # a bare 8-digit id: not a phone without spacing or +216
        "12345678901234",  # too long for any covered plan
        "1 234 567",  # a price: 7 digits
        "06 12 34 56",  # too short for France
    ],
)
def test_dates_ids_and_prices_are_not_phones(piece: str) -> None:
    assert not is_phone(piece)


def test_emails_and_phones_are_replaced_by_typed_placeholders() -> None:
    text = "Ecrivez a jean.dupont@example.fr ou appelez le 06 12 34 56 78."
    assert redact(text) == Redaction("Ecrivez a [EMAIL] ou appelez le [PHONE].", [EMAIL, PHONE])


def test_phone_inside_a_longer_number_is_left_alone() -> None:
    assert find_pii("code 9906123456789") == []  # glued digits: not a separate number


def test_nothing_found_leaves_the_text_untouched() -> None:
    text = "Le 4 septembre 2026, rien de personnel ici: prix 1 234 567,89 EUR."
    assert redact(text) == Redaction(text, [])


def test_spans_never_overlap_and_come_in_order() -> None:
    spans = find_pii("a@b.fr 06 12 34 56 78 c@d.com +216 71 234 567")
    assert [s.kind for s in spans] == [EMAIL, PHONE, EMAIL, PHONE]
    assert all(a.end <= b.start for a, b in zip(spans, spans[1:], strict=False))


def test_arabic_text_with_a_phone() -> None:
    text = "للاتصال: +216 71 234 567 شكرا"
    assert redact(text).text == "للاتصال: [PHONE] شكرا"
