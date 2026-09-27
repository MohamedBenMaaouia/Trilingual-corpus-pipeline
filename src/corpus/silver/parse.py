"""Silver step 2a: WET records become rows (Story 2.1, DECISIONS S2-04).

Two layers. parse_wet is pure Python: one conversion record in, one row out, and a
record that cannot be turned into a row becomes a dead letter instead of failing the
whole task. parse_files is the Spark wiring: it runs parse_wet inside the executors,
once per bronze file, and returns documents and dead letters in one DataFrame.
"""

import io
import uuid
from collections.abc import Iterable, Iterator
from datetime import datetime
from typing import IO, NamedTuple
from urllib.parse import urlsplit

import pandas as pd
from fastwarc import ArchiveIterator, WarcRecordType
from pyspark.sql import DataFrame
from pyspark.sql.types import (
    BinaryType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)
from tldextract import TLDExtract

from corpus.silver.normalize import normalize_common

# The Public Suffix List snapshot bundled with tldextract: no download (the default
# fetches publicsuffix.org once per process, i.e. per executor) and no cache written
# to a home folder. Built at import, so once per Python worker process.
_SUFFIXES = TLDExtract(suffix_list_urls=(), cache_dir=None)


class Document(NamedTuple):
    doc_id: str  # the record's WARC-Record-ID uuid (T9): unique, traceable to bronze
    url: str
    domain: str  # registrable domain, lowercase; never empty
    fetch_date: datetime  # WARC-Date, UTC
    content_length: int  # UTF-8 bytes, as the WARC header states (not characters)
    cc_language: str | None  # Common Crawl's own guess, e.g. "ara" or "fra,eng"
    text: str


class DeadLetter(NamedTuple):
    record_offset: int  # position of the record in the compressed file
    exception: str
    raw: bytes  # the record body, kept for diagnosis


def parse_wet(stream: IO[bytes]) -> Iterator[Document | DeadLetter]:
    """Yield one Document per conversion record in a WET stream, in file order.

    The stream is decompressed record by record, so a 180 MB file is never held
    as text all at once. warcinfo (file metadata) records are skipped.
    """
    for record in ArchiveIterator(stream, record_types=WarcRecordType.conversion):
        raw = record.reader.read()
        try:
            yield _to_document(record.headers, raw)
        except Exception as err:  # any bad record: keep it, never kill the task
            yield DeadLetter(record.stream_pos, f"{type(err).__name__}: {err}", raw)


def _to_document(headers: object, raw: bytes) -> Document:
    get = headers.get  # type: ignore[attr-defined]  # fastwarc's HeaderMap has no stubs
    content_length = int(get("Content-Length"))
    if len(raw) != content_length:  # a truncated record: the body is incomplete
        raise ValueError(f"body is {len(raw)} bytes, header says {content_length}")
    # "<urn:uuid:dbc0...>" -> "dbc0...". uuid.UUID rejects anything malformed.
    record_id = str(uuid.UUID(get("WARC-Record-ID").strip("<>").removeprefix("urn:uuid:")))
    url = get("WARC-Target-URI")
    return Document(
        doc_id=record_id,
        url=url,
        domain=registrable_domain(url),
        fetch_date=datetime.fromisoformat(get("WARC-Date")),  # "2026-09-04T14:40:25Z"
        content_length=content_length,
        cc_language=get("WARC-Identified-Content-Language"),
        text=raw.decode("utf-8"),  # strict: invalid bytes make a dead letter, not "?"
    )


def registrable_domain(url: str) -> str:
    """The site a URL belongs to: www.bbc.co.uk -> bbc.co.uk (Public Suffix List, ICANN part).

    Boilerplate detection groups documents by this value, so it must be stable:
    lowercased, because Example.COM and example.com are the same site. IP addresses,
    localhost and suffixes missing from the list have no registrable domain; they fall
    back to the bare hostname. No hostname at all raises, and the record becomes a
    dead letter: `domain` is never empty.
    """
    domain = _SUFFIXES(url).top_domain_under_public_suffix
    if domain:
        return domain.lower()
    host = urlsplit(url).hostname  # already lowercase, without port, user:pw@ or [ ]
    if not host:
        raise ValueError(f"no host in url {url!r}")
    return host


# What parse_files returns: one row per record. A document fills the document columns;
# a dead letter fills record_offset, exception and raw. Declared, never inferred.
PARSED_SCHEMA = StructType(
    [
        StructField("path", StringType(), False),  # the bronze file: lineage + dead letters
        StructField("doc_id", StringType(), True),
        StructField("url", StringType(), True),
        StructField("domain", StringType(), True),
        StructField("fetch_date", TimestampType(), True),
        StructField("content_length", LongType(), True),  # the raw record, as in bronze
        StructField("cc_language", StringType(), True),
        StructField("text", StringType(), True),  # after normalize_common (S2-06)
        StructField("record_offset", LongType(), True),
        StructField("exception", StringType(), True),  # null <=> this row is a document
        StructField("raw", BinaryType(), True),
    ]
)
_COLUMNS = PARSED_SCHEMA.fieldNames()
_OBJECT_COLUMNS = [
    f.name for f in PARSED_SCHEMA.fields if isinstance(f.dataType, StringType | BinaryType)
]

# Rows per pandas DataFrame handed back to Spark. Bounds executor memory: ~1,000
# documents of a few KB each, instead of a whole file's ~30,000 at once.
_ROWS_PER_CHUNK = 1_000


def parse_files(files: DataFrame) -> DataFrame:
    """Rows of Spark's binaryFile source (path, content) -> PARSED_SCHEMA rows.

    Document text comes out already through normalize_common. Runs in the executors
    through mapInPandas. The caller persists the result once and
    splits it on `exception`, so bronze is read and parsed a single time (S2-04).
    """
    return files.select("path", "content").mapInPandas(_parse_batches, schema=PARSED_SCHEMA)


def _parse_batches(batches: Iterable[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    """mapInPandas body: each input row is one whole compressed WET file."""
    for batch in batches:
        for path, content in zip(batch["path"], batch["content"], strict=True):
            chunk: list[dict[str, object]] = []
            for item in parse_wet(io.BytesIO(content)):
                if isinstance(item, Document):
                    # normalize_common here, not as a separate Spark step: the text is
                    # already in Python, so it costs no second JVM <-> Python transfer
                    # (S2-06). It must precede boilerplate line hashing (T10).
                    item = item._replace(text=normalize_common(item.text))
                chunk.append({"path": path, **item._asdict()})
                if len(chunk) == _ROWS_PER_CHUNK:
                    yield _to_frame(chunk)
                    chunk = []
            if chunk:
                yield _to_frame(chunk)


def _to_frame(chunk: list[dict[str, object]]) -> pd.DataFrame:
    """Rows -> a pandas DataFrame whose column types are pinned, never guessed.

    Documents and dead letters fill different fields, so a column can be empty for a
    whole chunk (`exception` is, when no record failed). pandas then guesses float64,
    and Spark 3.5's Arrow serializer fails converting it to a string column
    ("Expected Array, got ChunkedArray"). Same principle as the declared Spark schema.
    """
    frame = pd.DataFrame(chunk, columns=_COLUMNS)
    frame["fetch_date"] = pd.to_datetime(frame["fetch_date"], utc=True)  # all-empty -> NaT
    return frame.astype(
        {
            **{name: object for name in _OBJECT_COLUMNS},  # strings and bytes; nulls stay null
            "content_length": "Int64",  # pandas' nullable integer (capital I)
            "record_offset": "Int64",
        }
    )
