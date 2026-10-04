"""corpus.silver.language (Story 3.1, DECISIONS S3-01)."""

from collections.abc import Iterator, Sequence

import fasttext
import pytest
from pyspark import SparkFiles

from corpus.silver import language
from corpus.silver.language import (
    LANGUAGE_LOW_CONFIDENCE,
    LANGUAGE_NOT_TARGETED,
    LanguageResult,
    LanguageRules,
    Prediction,
    assign_language,
    predict_language,
)

RULES = LanguageRules()


class FakeModel:
    """Stands in for fastText: the real model is 126 MB and never enters the repo.
    Answers with a fixed label and probability, and records what it was asked."""

    def __init__(self, label: str = "en", prob: float = 0.9) -> None:
        self.label, self.prob = label, prob
        self.seen: list[str] = []

    def predict(self, text: str, k: int = 1) -> tuple[Sequence[str], Sequence[float]]:
        self.seen.append(text)
        return (f"__label__{self.label}",), (self.prob,)


@pytest.fixture(autouse=True)
def no_loaded_model() -> Iterator[None]:
    # The singleton is module state: every test starts, and leaves, without a model.
    language._loaded = None
    yield
    language._loaded = None


# Loading ------------------------------------------------------------------------------


def test_model_loads_once_per_process(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    loads: list[str] = []

    def fake_load(path: str) -> FakeModel:
        loads.append(path)
        return FakeModel()

    monkeypatch.setattr(SparkFiles, "get", lambda name: f"/executor/files/{name}")
    monkeypatch.setattr(fasttext, "load_model", fake_load)

    first = language.language_model()
    second = language.language_model()

    assert first is second  # the same object: no second load
    assert loads == ["/executor/files/lid.176.bin"]  # read once, from the executor's copy
    assert capsys.readouterr().err.count("language_model: loaded") == 1  # the evidence line


# Prediction ---------------------------------------------------------------------------


def test_newlines_become_spaces_before_predict() -> None:
    model = FakeModel()
    predict_language(model, "first line\nsecond line")
    assert model.seen == ["first line second line"]  # fastText rejects "\n"


def test_label_prefix_is_stripped() -> None:
    assert predict_language(FakeModel("arz", 0.8), "text") == Prediction("arz", 0.8)


def test_probability_above_one_is_capped() -> None:
    assert predict_language(FakeModel("en", 1.00001), "text") == Prediction("en", 1.0)


def test_empty_text_is_not_sent_to_the_model() -> None:
    model = FakeModel()
    assert predict_language(model, "") is None
    assert model.seen == []


# Assignment ---------------------------------------------------------------------------


@pytest.mark.parametrize("label", ["en", "fr", "ar"])
def test_confident_target_language_is_kept(label: str) -> None:
    assert assign_language(Prediction(label, 0.9), RULES) == LanguageResult(label, label, 0.9, None)


def test_threshold_itself_is_kept() -> None:
    # "lower than 0.65" is removed (FineWeb), so exactly 0.65 stays.
    assert assign_language(Prediction("fr", 0.65), RULES).reject_reason is None


def test_low_confidence_keeps_its_language() -> None:
    # Stays 'ar', so the breakdown shows low confidence per language (D10).
    assert assign_language(Prediction("ar", 0.4), RULES) == LanguageResult(
        "ar", "ar", 0.4, LANGUAGE_LOW_CONFIDENCE
    )


@pytest.mark.parametrize("label", ["de", "arz", "es"])
def test_other_languages_are_not_targeted(label: str) -> None:
    # arz (Egyptian Arabic) is not ar: the corpus is Modern Standard Arabic only.
    assert assign_language(Prediction(label, 0.99), RULES) == LanguageResult(
        "other", label, 0.99, LANGUAGE_NOT_TARGETED
    )


def test_empty_document_is_other_without_a_language_reason() -> None:
    assert assign_language(None, RULES) == LanguageResult("other", None, 0.0, None)


def test_thresholds_are_per_language() -> None:
    strict_arabic = LanguageRules(min_conf={"en": 0.65, "fr": 0.65, "ar": 0.8})
    assert assign_language(Prediction("ar", 0.7), strict_arabic).reject_reason == (
        LANGUAGE_LOW_CONFIDENCE
    )
    assert assign_language(Prediction("en", 0.7), strict_arabic).reject_reason is None
