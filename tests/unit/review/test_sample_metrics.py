"""corpus.review.sample_metrics: weighted numbers from labels (Story 3.6.4, S3-10).

Each expected value is worked out by hand in the comment beside it.
"""

from pathlib import Path

import pytest

from corpus.review.pii_sample import candidates, card_for
from corpus.review.sample_metrics import (
    lid_report,
    pii_report,
    read_labels,
    rejected_report,
    wilson,
)


def page(doc_id: str, stratum: str, detected: str, conf: float = 0.9) -> dict[str, object]:
    return {
        "doc_id": doc_id,
        "stratum": stratum,
        "language_detected": detected,
        "language_conf": conf,
    }


LID_RECORD = {
    "group_sizes": {"en_high": 1000, "en_missed": 100, "fr_high": 500},
    "pages": [
        page("a", "en_high", "en"),
        page("b", "en_high", "en"),
        page("c", "en_missed", "de"),
        page("d", "en_missed", "de"),
        page("e", "fr_high", "fr"),
        page("f", "fr_high", "fr"),
        page("unlabelled", "fr_high", "fr"),
    ],
}
LID_LABELS = {"a": "en", "b": "en", "c": "en", "d": "other", "e": "fr", "f": "en"}


def test_lid_precision_recall_and_accuracy_are_weighted_by_group_size() -> None:
    report = lid_report(LID_RECORD, LID_LABELS)
    assert report["labelled"] == 6
    assert report["per_language"]["en"]["precision"] == 1.0  # a, b: both en
    assert report["per_language"]["fr"]["precision"] == 0.5  # e fr, f en
    # True en pages: 1000 x 2/2 + 100 x 1/2 + 500 x 1/2 = 1300; kept as en: 1000.
    assert report["per_language"]["en"]["recall"] == pytest.approx(1000 / 1300, abs=1e-4)
    # fastText right: 1000 x 1 + 100 x 1/2 (d: "other" vs de) + 500 x 1/2 = 1300 of 1600.
    assert report["accuracy"] == pytest.approx(1300 / 1600, abs=1e-4)
    assert report["per_language"]["ar"]["precision"] is None  # no Arabic labelled


def test_pii_precision_and_weighted_recall() -> None:
    record = {
        "group_sizes": {"pii_caught": 1000, "pii_missed": 9000},
        "candidates": [
            {"candidate_id": "1", "stratum": "pii_caught"},
            {"candidate_id": "2", "stratum": "pii_caught"},
            {"candidate_id": "3", "stratum": "pii_missed"},
            {"candidate_id": "4", "stratum": "pii_missed"},
        ],
    }
    report = pii_report(record, {"1": "email", "2": "neither", "3": "phone", "4": "neither"})
    assert report["precision"] == 0.5
    # PII caught: 1000 x 1/2 = 500; PII missed: 9000 x 1/2 = 4500 -> recall 500 / 5000.
    assert report["recall"] == pytest.approx(0.1)


def test_rejected_precision_leaves_unsure_out() -> None:
    record = {
        "group_sizes": {"rejected_en": 900, "rejected_ar": 100},
        "pages": [
            {"doc_id": "a", "stratum": "rejected_en"},
            {"doc_id": "b", "stratum": "rejected_en"},
            {"doc_id": "c", "stratum": "rejected_en"},
            {"doc_id": "d", "stratum": "rejected_ar"},
            {"doc_id": "e", "stratum": "rejected_ar"},
        ],
    }
    labels = {"a": "junk", "b": "good", "c": "unsure", "d": "junk", "e": "junk"}
    report = rejected_report(record, labels)
    assert report["unsure"] == 1
    assert report["per_language"]["en"]["precision"] == 0.5
    assert report["per_language"]["ar"]["precision"] == 1.0
    assert report["precision"] == pytest.approx((900 * 0.5 + 100 * 1.0) / 1000)


def test_wilson_interval() -> None:
    assert wilson(0, 0) is None
    low, high = wilson(5, 10)  # type: ignore[misc]
    assert low == pytest.approx(0.2366, abs=1e-3) and high == pytest.approx(0.7634, abs=1e-3)


def test_labels_are_read_with_either_header(tmp_path: Path) -> None:
    old = tmp_path / "old.csv"
    old.write_text("doc_id,label\na,en\n", encoding="utf-8")  # the first LID page's header
    new = tmp_path / "new.csv"
    new.write_text("card_id,label\nx:3,phone\n", encoding="utf-8")
    assert read_labels(old) == {"a": "en"} and read_labels(new) == {"x:3": "phone"}


def test_pii_candidates_cover_what_the_redactor_catches_and_more() -> None:
    text = "Mail a.b@example.fr, tel 06 12 34 56 78, date 2026-09-04, id 98765432."
    found = {text[s:e]: caught for s, e, caught in candidates(text)}
    assert found["a.b@example.fr,"] is True  # "@" token (the comma comes with it)
    assert found["06 12 34 56 78"] is True
    assert found["2026-09-04"] is False  # a candidate the redactor rightly leaves alone
    assert found["98765432"] is False


def test_pii_card_highlights_the_piece_and_escapes_everything() -> None:
    text = "<b>x</b> call 06 12 34 56 78 now"
    card = card_for("d:14", text, 14, 28, 1)
    assert "<mark>06 12 34 56 78</mark>" in card.body_html
    assert "&lt;b&gt;x&lt;/b&gt;" in card.body_html
