"""Azure profile: Databricks + ADLS Gen2 (Sprint 10). Stub until then.

Roots will look like abfss://corpus-bronze@<storage-account>.dfs.core.windows.net.
No defaults yet: the storage account doesn't exist, so selecting this profile
without CORPUS_*_ROOT set fails at startup instead of pointing at nothing.
"""

from corpus.config.base import Settings


class AzureSettings(Settings):
    """Databricks owns the Spark session there: no master, no driver host, no S3 keys."""
