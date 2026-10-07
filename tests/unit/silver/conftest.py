"""Fixtures shared by the silver tests."""

import gzip
import uuid
from collections.abc import Callable

import pytest

WetRecord = Callable[..., bytes]


def _wet_record(
    body: bytes, record_id: str | None = None, url: str = "http://example.com/page"
) -> bytes:
    """One hand-made conversion record as its own gzip member, like a real WET file."""
    record_id = record_id or f"<urn:uuid:{uuid.uuid4()}>"
    headers = (
        "WARC/1.0\r\n"
        "WARC-Type: conversion\r\n"
        f"WARC-Target-URI: {url}\r\n"
        "WARC-Date: 2026-09-04T12:00:00Z\r\n"
        f"WARC-Record-ID: {record_id}\r\n"
        "Content-Type: text/plain\r\n"
        f"Content-Length: {len(body)}\r\n"
        "\r\n"
    )
    return gzip.compress(headers.encode() + body + b"\r\n\r\n")


@pytest.fixture
def wet_record() -> WetRecord:
    """Build WET records by hand, to make each failure case deliberate and visible."""
    return _wet_record
