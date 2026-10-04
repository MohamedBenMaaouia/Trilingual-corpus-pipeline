"""The only place storage paths are built. Callers never concatenate path strings.

Every path starts from a layer root in corpus.config, so the same call gives an
s3a:// path locally and an abfss:// path on Azure. Builders are added in the
sprint that first uses them (silver_path, gold_path, signature_store_path later).
"""

import re

from corpus.config import get_settings

# Common Crawl crawl ids look like CC-MAIN-2026-18 (year, then week of the year).
_CRAWL_ID = re.compile(r"CC-MAIN-\d{4}-\d{2}")


def check_crawl_id(crawl_id: str) -> None:
    # A bad id would silently create a new, wrong folder instead of failing.
    if not _CRAWL_ID.fullmatch(crawl_id):
        raise ValueError(f"not a Common Crawl id: {crawl_id!r} (expected CC-MAIN-YYYY-WW)")


def bronze_path(crawl_id: str, segment: int) -> str:
    """Folder holding one downloaded WET file, e.g.
    s3a://corpus-bronze/common_crawl/crawl_id=CC-MAIN-2026-18/segment=00042

    segment = index of the file in the crawl's wet.paths list, 5 digits (T6).
    """
    check_crawl_id(crawl_id)
    if not 0 <= segment <= 99_999:
        raise ValueError(f"segment must be 0..99999, got {segment}")
    root = get_settings().bronze_root
    return f"{root}/common_crawl/crawl_id={crawl_id}/segment={segment:05d}"


def bronze_object(crawl_id: str, segment: int, filename: str) -> str:
    """The final object for one downloaded WET file, name kept as published."""
    if "/" in filename or filename in ("", ".", ".."):
        # The filename comes from the crawl manifest: never let it climb out of its folder.
        raise ValueError(f"not a plain file name: {filename!r}")
    return f"{bronze_path(crawl_id, segment)}/{filename}"


def bronze_tmp_object(crawl_id: str, segment: int, filename: str) -> str:
    """Where an upload lands before it is promoted (T7). Nothing reads under _tmp."""
    if "/" in filename or filename in ("", ".", ".."):
        raise ValueError(f"not a plain file name: {filename!r}")
    return f"{bronze_path(crawl_id, segment)}/_tmp/{filename}"


def split_s3a(uri: str) -> tuple[str, str]:
    """s3a://bucket/some/key -> ("bucket", "some/key"), which is what boto3 wants."""
    if not uri.startswith("s3a://"):
        raise ValueError(f"not an s3a uri: {uri!r}")
    bucket, _, key = uri.removeprefix("s3a://").partition("/")
    if not bucket or not key:
        raise ValueError(f"uri has no bucket or no key: {uri!r}")
    return bucket, key


def _layer_root(root: str, dev: bool) -> str:
    """Dev runs write under <root>/_dev, never beside full runs (T11). The leading "_"
    makes Spark treat the folder as hidden when it discovers a table's partitions."""
    return f"{root}/_dev" if dev else root


def silver_stage1_path(crawl_id: str, *, dev: bool) -> str:
    """Sprint 2's interim silver output for one crawl (plan 2.4.1, DECISIONS S2-10), e.g.
    s3a://corpus-silver/_stage1/crawl_id=CC-MAIN-2026-39

    Crawl-scoped: overwriting it replaces that crawl and nothing else (invariant 6).
    """
    check_crawl_id(crawl_id)
    return f"{_layer_root(get_settings().silver_root, dev)}/_stage1/crawl_id={crawl_id}"


def silver_v1_path(*, dev: bool) -> str:
    """The silver table, all crawls (plan 3.5.1, DECISIONS S3-09), e.g.
    s3a://corpus-silver/silver_v1/language=ar/quality_tier=high/crawl_id=CC-MAIN-2026-39

    Partitioned by language, quality tier, crawl. Never overwritten as a whole: a rerun
    replaces one crawl's folders only (D18, silver_v1_crawl_glob)."""
    return f"{_layer_root(get_settings().silver_root, dev)}/silver_v1"


def silver_v1_crawl_glob(crawl_id: str, *, dev: bool) -> str:
    """Every folder of one crawl in silver_v1, as a glob: what a rerun deletes (D18)."""
    check_crawl_id(crawl_id)
    return f"{silver_v1_path(dev=dev)}/language=*/quality_tier=*/crawl_id={crawl_id}"


# Stages that produce dead letters. Checked, so a typo cannot create a new folder.
_DEAD_LETTER_STAGES = frozenset({"silver"})


def dead_letter_path(stage: str, crawl_id: str, *, dev: bool) -> str:
    """Records a stage could not process, kept for diagnosis (T1: dead letters in meta), e.g.
    s3a://corpus-meta/dead_letter/silver/crawl_id=CC-MAIN-2026-39"""
    if stage not in _DEAD_LETTER_STAGES:
        raise ValueError(f"unknown stage {stage!r}, expected one of {sorted(_DEAD_LETTER_STAGES)}")
    check_crawl_id(crawl_id)
    root = _layer_root(get_settings().meta_root, dev)
    return f"{root}/dead_letter/{stage}/crawl_id={crawl_id}"


def lid_model_path() -> str:
    """fastText's language ID model (plan 3.1.1): uploaded by hand, never committed, e.g.
    s3a://corpus-meta/models/lid.176.bin

    The job ships it to the executors with addFile, which keeps the file name, so it
    must stay "lid.176.bin" (silver.language.MODEL_FILE; a test ties the two). Not
    dev-scoped: dev and full runs use the same model.
    """
    return f"{get_settings().meta_root}/models/lid.176.bin"


def spark_events_path() -> str:
    """Spark event logs, read by the history server (placeholder .keep made by minio-init)."""
    return f"{get_settings().meta_root}/spark-events"
