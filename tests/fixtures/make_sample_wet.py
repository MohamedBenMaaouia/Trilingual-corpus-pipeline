"""Cut tests/fixtures/sample.wet.gz out of one real Common Crawl WET file (task 0.4.2).

Keeps the file's warcinfo record plus the first English, French and Arabic
conversion records of a reasonable size, so the fixture is small (~50 KB)
but real. Each record is written as its own gzip member, like real WET files.

    python tests/fixtures/make_sample_wet.py <input.warc.wet.gz> tests/fixtures/sample.wet.gz
"""

import gzip
import sys
from collections.abc import Iterator
from io import BufferedIOBase

WANTED = ("eng", "fra", "ara")  # primary language as Common Crawl identified it
MIN_BYTES, MAX_BYTES = 2_000, 15_000  # skip near-empty pages and huge ones


def read_records(stream: BufferedIOBase) -> Iterator[tuple[dict[str, str], bytes]]:
    """Yield (headers, full raw record bytes) for each WARC record in a stream."""
    while True:
        version = stream.readline()
        if not version:
            return
        if not version.strip():
            continue  # blank lines between records
        header_lines = [version]
        headers: dict[str, str] = {}
        while (line := stream.readline()).strip():
            header_lines.append(line)
            name, _, value = line.decode("utf-8").partition(":")
            headers[name.strip()] = value.strip()
        content = stream.read(int(headers["Content-Length"]))
        yield headers, b"".join(header_lines) + b"\r\n" + content + b"\r\n\r\n"


def main(src: str, dst: str) -> None:
    chosen: dict[str, tuple[dict[str, str], bytes]] = {}
    warcinfo: bytes | None = None
    with gzip.open(src, "rb") as stream:  # reads across all gzip members
        for headers, raw in read_records(stream):
            if headers["WARC-Type"] == "warcinfo":
                warcinfo = raw
                continue
            lang = headers.get("WARC-Identified-Content-Language", "").split(",")[0]
            size = int(headers["Content-Length"])
            if lang in WANTED and lang not in chosen and MIN_BYTES <= size <= MAX_BYTES:
                chosen[lang] = (headers, raw)
            if len(chosen) == len(WANTED):
                break
    missing = set(WANTED) - set(chosen)
    if warcinfo is None or missing:
        sys.exit(f"not found in {src}: {sorted(missing) or 'warcinfo'}")

    with open(dst, "wb") as out:
        out.write(gzip.compress(warcinfo))
        for lang in WANTED:
            out.write(gzip.compress(chosen[lang][1]))

    for lang in WANTED:
        headers, _ = chosen[lang]
        print(f"{lang}  {headers['Content-Length']:>6} bytes  {headers['WARC-Target-URI']}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
