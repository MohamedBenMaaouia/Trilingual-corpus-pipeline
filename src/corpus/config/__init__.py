"""The one entry point to configuration: get_settings().

CORPUS_ENV picks the profile: local (default, Docker) or azure (Sprint 10).
"""

import os
from functools import lru_cache

from corpus.config.azure import AzureSettings
from corpus.config.base import Settings
from corpus.config.local import LocalSettings

_PROFILES: dict[str, type[Settings]] = {
    "local": LocalSettings,
    "azure": AzureSettings,
}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load the active profile once per process; later calls return the same object."""
    env = os.environ.get("CORPUS_ENV", "local")
    if env not in _PROFILES:
        # A typo like "lcoal" must not silently run against the wrong environment.
        raise ValueError(f"CORPUS_ENV={env!r} is not one of {sorted(_PROFILES)}")
    return _PROFILES[env]()
