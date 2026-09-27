"""corpus.silver.parse: WET records to rows, bad records to dead letters (task 2.1.4)."""

import io
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from corpus.silver import parse
from corpus.silver.parse import DeadLetter, Document, parse_wet, registrable_domain

SAMPLE = Path(__file__).parents[2] / "fixtures" / "sample.wet.gz"

WetRecord = Callable[..., bytes]  # the `wet_record` fixture, from conftest.py


def parse_sample() -> list[Document | DeadLetter]:
    with SAMPLE.open("rb") as stream:
        return list(parse_wet(stream))


def test_fixture_gives_exactly_the_three_documents_in_file_order() -> None:
    rows = parse_sample()
    assert all(isinstance(row, Document) for row in rows)  # no dead letters, no warcinfo
    assert [row.url for row in rows if isinstance(row, Document)] == [
        "http://01xq.com/xqplayer/xqplayer.asp?pid=110764",
        "http://comblan.com/village-martyr-comblanchien/photos-1945/vue-du-village-detruit/",
        "http://www.7awaya.com/poetPage.do;jsessionid=DD8B5DC82D9B8BFCEEBBE5AA03358BB1?poetId=365",
    ]


def test_every_field_of_a_known_record() -> None:
    first = parse_sample()[0]
    assert isinstance(first, Document)
    assert first.doc_id == "dbc0bc3a-c7db-4a16-a82d-420a7eb85610"  # no <urn:uuid:...> wrapper
    assert first.domain == "01xq.com"
    assert first.fetch_date == datetime(2026, 9, 4, 14, 40, 25, tzinfo=UTC)
    assert first.content_length == 3496
    assert first.cc_language == "eng"
    assert first.text.startswith("ChenYiFan Introduce - XiangQi Database")


def test_content_length_counts_bytes_not_characters() -> None:
    arabic = parse_sample()[2]
    assert isinstance(arabic, Document)
    assert arabic.cc_language == "ara"
    assert arabic.content_length == 3920  # UTF-8 bytes: Arabic letters take 2 each
    assert len(arabic.text) == 2287  # characters


def test_a_bad_record_becomes_a_dead_letter_and_parsing_continues(wet_record: WetRecord) -> None:
    bad_body = b"caf\xe9"  # Latin-1 byte, not valid UTF-8
    stream = io.BytesIO(
        wet_record(b"first good record") + wet_record(bad_body) + wet_record(b"second good record")
    )
    rows = list(parse_wet(stream))
    assert [type(row) for row in rows] == [Document, DeadLetter, Document]
    dead = rows[1]
    assert isinstance(dead, DeadLetter)
    assert dead.exception.startswith("UnicodeDecodeError")
    assert dead.raw == bad_body  # kept, so the record can be inspected later
    assert dead.record_offset > 0  # where it starts in the compressed stream


def test_a_malformed_record_id_becomes_a_dead_letter(wet_record: WetRecord) -> None:
    rows = list(parse_wet(io.BytesIO(wet_record(b"text", record_id="<urn:uuid:not-a-uuid>"))))
    assert len(rows) == 1
    assert isinstance(rows[0], DeadLetter)
    assert rows[0].exception.startswith("ValueError")


@pytest.mark.parametrize(
    ("url", "domain"),
    [
        ("http://www.bbc.co.uk/news", "bbc.co.uk"),  # two-part public suffix
        ("https://Example.COM:8080/a", "example.com"),  # lowercased, port dropped
        ("http://user:pw@shop.example.fr/", "example.fr"),  # credentials ignored
        ("http://myblog.blogspot.com/", "blogspot.com"),  # ICANN rules only
        ("http://192.168.1.10/page", "192.168.1.10"),  # IP: falls back to the host
        ("http://[2001:db8::1]/p", "2001:db8::1"),  # IPv6, brackets dropped
        ("http://localhost:8000/", "localhost"),
        ("http://xn--mgbh0fb.xn--kgbechtv/", "xn--mgbh0fb.xn--kgbechtv"),  # suffix not listed
    ],
)
def test_registrable_domain(url: str, domain: str) -> None:
    assert registrable_domain(url) == domain


def test_the_suffix_list_never_comes_from_the_network() -> None:
    """The bundled snapshot only: no download on every executor (trap S2)."""
    assert parse._SUFFIXES.suffix_list_urls == ()


def test_a_url_without_a_host_becomes_a_dead_letter(wet_record: WetRecord) -> None:
    rows = list(parse_wet(io.BytesIO(wet_record(b"text", url="not a url"))))
    assert len(rows) == 1
    assert isinstance(rows[0], DeadLetter)
    assert "no host" in rows[0].exception
