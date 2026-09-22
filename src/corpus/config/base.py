"""Settings shared by every execution profile (local, azure).

Only corpus.config reads environment variables. Values come from the process
environment (docker compose injects them from .env); nothing here reads .env itself.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """What every job needs to know, whatever the environment.

    A field `foo` is read from the environment variable CORPUS_FOO.
    Fields are added in the sprint that first uses them.
    """

    model_config = SettingsConfigDict(env_prefix="CORPUS_", frozen=True)

    # One storage root per layer (T1). Paths below a root come from corpus.io only.
    bronze_root: str
    silver_root: str
    gold_root: str
    meta_root: str
