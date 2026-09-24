"""Stage 1: fetch bronze segments over HTTPS, validate them, store them.

Plain Python, not Spark: 50 files is network waiting, not computation.
Nothing here looks inside a WET file (invariant 1: bronze is raw and immutable).
"""

import gzip
import hashlib
import re
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from corpus.acquisition.control import SegmentControl
from corpus.acquisition.manifest import Segment
from corpus.io import bronze_object, bronze_tmp_object
from corpus.store import promote, put_file

# Identify this client to a free public service instead of "python-requests/2.x".
USER_AGENT = "corpus-pipeline/0.1 (trilingual web corpus, educational project)"

# (connect, read). Read timeout applies between chunks, not to the whole download,
# so a big file is fine but a hung connection is not.
TIMEOUT_SECONDS = (10, 60)

# data.commoncrawl.org answers 503 "SlowDown" under load; 429 is the standard
# "too many requests". A 404 is not retried: the file is simply not there.
RETRY_STATUSES = (429, 500, 502, 503, 504)

# Read and decompress in 4 MiB pieces: never hold a whole file in memory.
CHUNK_BYTES = 4 * 1024 * 1024

# A real WET segment decompresses to ~180 MB. This cap (about 20x that) stops a
# compression bomb from filling the disk (DECISIONS S1-03).
MAX_DECOMPRESSED_BYTES = 4 * 1024**3

# 4 parallel downloads: polite to a free public service, and enough for a home
# connection. Downloading is waiting on the network, not computing.
DEFAULT_THREADS = 4

# A segment takes under a minute, so a claim older than this belonged to a process
# that died, and a restart may take it back (T8).
STALE_LEASE_SECONDS = 15 * 60

# A segment gets this many tries in total before it stays failed for a human to see.
MAX_ATTEMPTS = 3

# One bad segment must not waste the others; a tenth of them failing means something
# systemic (network, rate limit, disk), and hammering on would make it worse.
FAILURE_RATE_ABORT = 0.10


# An S3 object uploaded in one part has an ETag that IS the MD5 of its contents.
# A multipart upload looks like "hash-12": a hash of hashes, not comparable.
_MD5_ETAG = re.compile(r'"?([0-9a-f]{32})"?')


class DownloadError(Exception):
    """A segment could not be fetched or did not pass validation."""


def http_session(*, total_retries: int = 5, backoff_factor: float = 1.0) -> requests.Session:
    """A session that retries temporary failures with growing pauses.

    The retry policy lives in the adapter, so every request made through this
    session gets it and no calling code needs retry logic of its own.
    backoff_factor is a parameter so tests do not sleep.
    """
    retry = Retry(
        total=total_retries,
        backoff_factor=backoff_factor,  # pauses ~1, 2, 4, 8, 16 s at factor 1
        status_forcelist=RETRY_STATUSES,
        allowed_methods=frozenset(["GET", "HEAD"]),  # only repeat safe requests
        respect_retry_after_header=True,  # obey the server's own "wait n seconds"
        raise_on_status=False,  # let callers use response.raise_for_status()
    )
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    adapter = HTTPAdapter(max_retries=retry)
    # Production is https only; http is mounted too so the policy is the same
    # everywhere, including for the local test server.
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


class RemoteFile(NamedTuple):
    """What the server says about a file, before we fetch it."""

    size_bytes: int | None  # Content-Length, the number validate_size checks against
    etag: str | None  # recorded for reference; not a checksum we can verify (C2)
    resumable: bool  # server advertises Accept-Ranges: bytes


def head(session: requests.Session, url: str) -> RemoteFile:
    """Ask about the file without downloading it."""
    response = session.head(url, timeout=TIMEOUT_SECONDS, allow_redirects=True)
    response.raise_for_status()
    length = response.headers.get("Content-Length")
    return RemoteFile(
        size_bytes=int(length) if length is not None else None,
        etag=response.headers.get("ETag"),
        resumable=response.headers.get("Accept-Ranges", "").lower() == "bytes",
    )


def stream_to_file(
    session: requests.Session, url: str, destination: Path, *, resumable: bool = False
) -> int:
    """Download to a local file in chunks. Returns the file's total size in bytes.

    If a partial file is already there and the server allows ranges, the download
    continues from that point (T7: only a local file can be appended to).
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    already_have = destination.stat().st_size if destination.exists() else 0

    headers = {}
    if already_have and resumable:
        headers["Range"] = f"bytes={already_have}-"  # "send me the rest"

    with session.get(url, stream=True, timeout=TIMEOUT_SECONDS, headers=headers) as response:
        response.raise_for_status()
        if headers and response.status_code == 206:
            mode = "ab"  # 206 Partial Content: the server honoured the range, append
        else:
            # No range asked, or the server ignored it and is sending the whole file:
            # start the file over, or we would splice two copies together.
            mode, already_have = "wb", 0
        with destination.open(mode) as out:
            for chunk in response.iter_content(CHUNK_BYTES):
                out.write(chunk)
                already_have += len(chunk)
    return already_have


class SegmentResult(NamedTuple):
    """What one successful download established, for the control table to record."""

    segment: Segment
    size_bytes: int
    etag: str | None
    checksum_verified: bool  # False = the ETag was not comparable, not "wrong"
    decompressed_bytes: int
    final_uri: str


def staging_path(staging_dir: Path, crawl_id: str, segment: Segment) -> Path:
    """Local file a download is written into: one per crawl and segment."""
    return staging_dir / crawl_id / f"{segment.segment_id}.warc.wet.gz"


def download_segment(
    session: requests.Session,
    crawl_id: str,
    segment: Segment,
    *,
    staging_dir: Path,
    max_decompressed_bytes: int = MAX_DECOMPRESSED_BYTES,
    source_url: str | None = None,  # defaults to the real one; tests point it elsewhere
) -> SegmentResult:
    """Fetch, validate and store one segment. Bronze sees nothing unvalidated.

    On a validation failure the staging file is deleted: it cannot be trusted as a
    base for a resumed download. On a network failure it is kept, which is exactly
    what resume is for.
    """
    url = source_url or segment.source_url
    remote = head(session, url)
    local = staging_path(staging_dir, crawl_id, segment)

    stored = stream_to_file(session, url, local, resumable=remote.resumable)
    try:
        validate_size(stored, remote.size_bytes)
        checksum_verified = verify_etag(local, remote.etag)
        decompressed = validate_gzip(local, max_decompressed_bytes=max_decompressed_bytes)
    except DownloadError:
        local.unlink(missing_ok=True)
        raise

    filename = segment.path.rsplit("/", 1)[-1]  # keep the published name (invariant 1)
    temp_uri = bronze_tmp_object(crawl_id, int(segment.segment_id), filename)
    final_uri = bronze_object(crawl_id, int(segment.segment_id), filename)
    put_file(local, temp_uri)
    promote(temp_uri, final_uri)
    local.unlink(missing_ok=True)  # the local copy has served its purpose

    return SegmentResult(
        segment=segment,
        size_bytes=stored,
        etag=remote.etag,
        checksum_verified=checksum_verified,
        decompressed_bytes=decompressed,
        final_uri=final_uri,
    )


class RunSummary(NamedTuple):
    """What one download run achieved. Story 1.6 turns this into run_metrics rows."""

    complete: int
    failed: int
    bytes_downloaded: int
    duration_seconds: float


def run_download(
    control: SegmentControl,
    session: requests.Session,
    crawl_id: str,
    *,
    staging_dir: Path,
    threads: int = DEFAULT_THREADS,
    max_decompressed_bytes: int = MAX_DECOMPRESSED_BYTES,
    base_url: str | None = None,  # tests point this at a local server
) -> RunSummary:
    """Download every pending segment of a crawl. Safe to kill and restart at any point.

    Only this thread talks to Postgres (a psycopg2 connection is not thread-safe);
    the worker threads only fetch files.
    """
    started = time.monotonic()
    # First, before anything else: rows a dead process left in 'downloading' would
    # otherwise never be picked up again, and a restart would find nothing to do.
    control.reclaim_stale(crawl_id, STALE_LEASE_SECONDS)
    control.retry_failed(crawl_id, MAX_ATTEMPTS)

    intended = sum(control.status_counts(crawl_id).values())
    complete = failed = 0
    bytes_downloaded = 0

    with ThreadPoolExecutor(max_workers=threads) as pool:
        while True:
            # Claim in small batches: a crash leaves at most `threads` rows to reclaim.
            batch = control.claim_pending(crawl_id, limit=threads)
            if not batch:
                break
            running = {
                pool.submit(
                    download_segment,
                    session,
                    crawl_id,
                    segment,
                    staging_dir=staging_dir,
                    max_decompressed_bytes=max_decompressed_bytes,
                    source_url=f"{base_url}/{segment.path}" if base_url else None,
                ): segment
                for segment in batch
            }
            for future in as_completed(running):
                segment = running[future]
                try:
                    result = future.result()
                except Exception as err:  # network, validation or storage
                    control.mark_failed(
                        crawl_id, segment.segment_id, f"{type(err).__name__}: {err}"
                    )
                    failed += 1
                    outcome = f"FAILED ({type(err).__name__})"
                else:
                    control.mark_complete(
                        crawl_id,
                        segment.segment_id,
                        size_bytes=result.size_bytes,
                        etag=result.etag,
                        final_key=result.final_uri,
                    )
                    complete += 1
                    bytes_downloaded += result.size_bytes
                    verified = "checksum ok" if result.checksum_verified else "checksum n/a"
                    outcome = f"ok, {result.size_bytes / 1e6:.0f} MB, {verified}"
                # One line per segment, so an Airflow log shows progress instead of
                # nothing for twenty minutes. flush: it appears as it happens.
                print(
                    f"[{complete + failed}/{intended}] segment {segment.segment_id}: {outcome}",
                    flush=True,
                )

            if failed > FAILURE_RATE_ABORT * intended:
                raise DownloadError(
                    f"{failed} of {intended} segments failed "
                    f"(over {FAILURE_RATE_ABORT:.0%}): stopping the run"
                )

    return RunSummary(
        complete=complete,
        failed=failed,
        bytes_downloaded=bytes_downloaded,
        duration_seconds=time.monotonic() - started,
    )


def validate_size(received: int, expected: int | None) -> None:
    """The bytes we stored must be exactly what the server promised.

    Catches a connection cut mid-file. Common Crawl publishes no checksums, so this
    plus validate_gzip is the whole of our integrity check (C2).
    """
    if expected is None:
        raise DownloadError("server sent no Content-Length: cannot verify the download")
    if received != expected:
        raise DownloadError(f"size mismatch: stored {received} bytes, expected {expected}")


def md5_of_file(path: Path) -> str:
    """MD5 of the file's bytes. Used for integrity against S3 ETags, not for security."""
    digest = hashlib.md5()  # noqa: S324 - integrity check against the server's ETag
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def verify_etag(path: Path, etag: str | None) -> bool:
    """Compare the file with the server's ETag when that is possible.

    Returns True if the ETag was a plain MD5 and matched, False if it could not be
    compared (missing, or a multipart "hash-N" ETag). Raises DownloadError on a
    mismatch. The caller records which of the two happened, so we never claim a
    checksum was verified when it was not (C2).
    """
    match = _MD5_ETAG.fullmatch(etag or "")
    if match is None:
        return False
    actual = md5_of_file(path)
    if actual != match.group(1):
        raise DownloadError(f"checksum mismatch for {path.name}: {actual} != {match.group(1)}")
    return True


def validate_gzip(path: Path, *, max_decompressed_bytes: int = MAX_DECOMPRESSED_BYTES) -> int:
    """Decompress the file completely, discarding the data. Returns decompressed size.

    Proves completeness: gzip ends each member with a checksum and a length, so a
    truncated or corrupted file fails here even when its byte count looked right.
    WET files are multi-member gzip (one member per record), and gzip.open reads
    through all members, so this reaches the real end of the file.

    Also enforces the decompression cap: it stops as soon as the output grows past
    max_decompressed_bytes, instead of writing a compression bomb through memory.
    """
    total = 0
    try:
        with gzip.open(path, "rb") as stream:
            while chunk := stream.read(CHUNK_BYTES):
                total += len(chunk)
                if total > max_decompressed_bytes:
                    raise DownloadError(
                        f"{path.name}: decompressed size passed "
                        f"{max_decompressed_bytes} bytes, refusing it"
                    )
    except (OSError, EOFError, zlib.error) as err:  # BadGzipFile is an OSError
        raise DownloadError(f"{path.name} is not a complete gzip file: {err}") from err
    return total
