"""The one entry point to configuration: get_settings().

CORPUS_ENV picks the profile: local (default, Docker) or azure (Sprint 10).
"""

import os
from functools import lru_cache

from pydantic import ValidationError

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
    try:
        return _PROFILES[env]()
    except ValidationError as err:
        # Pydantic's message quotes the values it received, which would print secrets
        # into task logs. Report the field names and the reasons only.
        problems = ", ".join(
            f"CORPUS_{e['loc'][0]}".upper() + f" ({e['msg']})" for e in err.errors()
        )
        raise ValueError(f"bad configuration for CORPUS_ENV={env}: {problems}") from None
