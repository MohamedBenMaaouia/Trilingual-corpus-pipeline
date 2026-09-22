"""The only place storage paths are built. Callers never concatenate path strings.

Every path starts from a layer root in corpus.config, so the same call gives an
s3a:// path locally and an abfss:// path on Azure. Builders are added in the
sprint that first uses them (silver_path, gold_path, signature_store_path later).
"""

import re

from corpus.config import get_settings

# Common Crawl crawl ids look like CC-MAIN-2026-18 (year, then week of the year).
_CRAWL_ID = re.compile(r"CC-MAIN-\d{4}-\d{2}")


def _check_crawl_id(crawl_id: str) -> None:
    # A bad id would silently create a new, wrong folder instead of failing.
    if not _CRAWL_ID.fullmatch(crawl_id):
        raise ValueError(f"not a Common Crawl id: {crawl_id!r} (expected CC-MAIN-YYYY-WW)")


def bronze_path(crawl_id: str, segment: int) -> str:
    """Folder holding one downloaded WET file, e.g.
    s3a://corpus-bronze/common_crawl/crawl_id=CC-MAIN-2026-18/segment=00042

    segment = index of the file in the crawl's wet.paths list, 5 digits (T6).
    """
    _check_crawl_id(crawl_id)
    if not 0 <= segment <= 99_999:
        raise ValueError(f"segment must be 0..99999, got {segment}")
    root = get_settings().bronze_root
    return f"{root}/common_crawl/crawl_id={crawl_id}/segment={segment:05d}"


def smoke_output_path() -> str:
    """Sprint 0 smoke job output (task 0.5.2)."""
    return f"{get_settings().silver_root}/smoke"


def spark_events_path() -> str:
    """Spark event logs, read by the history server (placeholder .keep made by minio-init)."""
    return f"{get_settings().meta_root}/spark-events"
