"""corpus.config: profile selection, required secrets, masking, immutability."""

import os
from collections.abc import Iterator

import pytest
from pydantic import ValidationError

from corpus.config import get_settings
from corpus.config.azure import AzureSettings
from corpus.config.local import LocalSettings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Each test starts with no CORPUS_* variables and no cached settings."""
    for name in list(os.environ):
        if name.startswith("CORPUS_"):
            monkeypatch.delenv(name)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def s3_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORPUS_S3_ACCESS_KEY", "test-user")
    monkeypatch.setenv("CORPUS_S3_SECRET_KEY", "test-secret")


def test_default_profile_is_local(s3_keys: None) -> None:
    assert isinstance(get_settings(), LocalSettings)


def test_local_roots_are_the_four_minio_buckets(s3_keys: None) -> None:
    s = get_settings()
    assert (s.bronze_root, s.silver_root, s.gold_root, s.meta_root) == (
        "s3a://corpus-bronze",
        "s3a://corpus-silver",
        "s3a://corpus-gold",
        "s3a://corpus-meta",
    )


def test_env_variable_overrides_a_default(s3_keys: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORPUS_SILVER_ROOT", "s3a://other-silver")
    assert get_settings().silver_root == "s3a://other-silver"


def test_missing_s3_keys_fail_at_startup() -> None:
    with pytest.raises(ValidationError) as err:
        get_settings()
    missing = {e["loc"][0] for e in err.value.errors()}
    assert missing == {"s3_access_key", "s3_secret_key"}


def test_secret_is_masked_when_printed(s3_keys: None) -> None:
    s = get_settings()
    assert "test-secret" not in repr(s)
    assert isinstance(s, LocalSettings)
    assert s.s3_secret_key.get_secret_value() == "test-secret"


def test_settings_are_frozen(s3_keys: None) -> None:
    with pytest.raises(ValidationError):
        get_settings().gold_root = "s3a://elsewhere"  # type: ignore[misc]


def test_settings_are_loaded_once_per_process(s3_keys: None) -> None:
    assert get_settings() is get_settings()


def test_azure_profile_requires_its_roots(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORPUS_ENV", "azure")
    with pytest.raises(ValidationError):
        get_settings()


def test_azure_profile_is_selected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORPUS_ENV", "azure")
    for layer in ("bronze", "silver", "gold", "meta"):
        monkeypatch.setenv(f"CORPUS_{layer.upper()}_ROOT", f"abfss://corpus-{layer}@acct.dfs.core.windows.net")
    assert isinstance(get_settings(), AzureSettings)


def test_unknown_profile_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORPUS_ENV", "lcoal")
    with pytest.raises(ValueError, match="lcoal"):
        get_settings()
