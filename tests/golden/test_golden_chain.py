"""The golden integration test (Story 7.5.1, DECISIONS S7-06).

100 made-up pages, each built to land in a known place, go through the whole transform
chain on local files, exactly as the jobs run it: WET files -> silver stage 1 (parse,
boilerplate, normalization) -> silver_v1 (language ID, PII, quality) -> dedup -> gold
(Delta, with an excluded site). Each page's outcome is checked twice:
- against the intent it was built for (readable here, page by page);
- against tests/golden/expected.json, the committed outcome of every page, down to a hash
  of its gold text: a change anywhere in the chain that moves a page shows up here, even
  when every unit test still passes.

The pages are generated from tests/fixtures/lid_vocabulary.json, the words the test
language model is trained on (tests/conftest.py): no page of the web is reproduced.

After an intended change, rewrite expected.json (the repo is mounted read-only, so the
folder is mounted again, writable) and review its diff:
    docker compose run --rm -v ./tests/golden:/golden -e GOLDEN_WRITE=/golden/expected.json
        tests pytest tests/golden
"""

import gzip
import hashlib
import json
import os
import random
import uuid
import zlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from corpus.jobs.run_dedup import DedupParams, run_dedup
from corpus.jobs.run_gold import run_gold
from corpus.jobs.run_silver import build_stage1
from corpus.jobs.run_silver_v1 import run_silver_v1
from corpus.schemas.dedup_v1 import DEDUP_V1
from corpus.schemas.silver_stage1 import STAGE1_SCHEMA
from corpus.schemas.silver_v1 import SILVER_V1

HERE = Path(__file__).parent
EXPECTED = HERE / "expected.json"
VOCABULARY = json.loads(
    (HERE.parent / "fixtures" / "lid_vocabulary.json").read_text(encoding="utf-8")
)

CRAWL = "CC-MAIN-2026-39"
EXCLUDED_SITE = "excluded-golden.com"
LF, CRLF = chr(10), chr(13) + chr(10)
FATHA = chr(0x064E)  # an Arabic vowel mark: arabic_match_key removes it (D6)
NAMESPACE = uuid.UUID("6f1c9c2e-1d6a-4f53-9a8e-2b7f0c3d4e5a")  # fixed: stable doc ids


@dataclass(frozen=True)
class Intent:
    """Where a page is built to land."""

    tier: str  # high | medium | rejected
    language: str | None  # en | fr | ar | other; None: no clear winner by design
    reason: str | None = None  # a reason it must be rejected for
    duplicate: str | None = None  # exact | near: dedup must flag it
    gold: bool = False  # must be in gold
    pii: tuple[str, ...] = ()  # PII types it must have had redacted


@dataclass(frozen=True)
class Page:
    key: str  # stable, readable name; the doc_id derives from it
    url: str
    day: int  # fetch date: 2026-09-<day>
    lines: list[str]
    intent: Intent

    @property
    def doc_id(self) -> str:
        return str(uuid.uuid5(NAMESPACE, self.key))


@dataclass
class Builder:
    """Seeded sentences from the vocabulary. Each page draws from its own generator,
    seeded from its key (zlib.crc32, never hash(): it changes between processes)."""

    pages: list[Page] = field(default_factory=list)

    @staticmethod
    def rng(key: str) -> random.Random:
        return random.Random(zlib.crc32(key.encode("utf-8")))

    @staticmethod
    def sentence(rng: random.Random, lang: str, words: int, stop_share: float) -> str:
        vocab = VOCABULARY[lang]
        picked = [
            rng.choice(vocab["stopwords"])
            if rng.random() < stop_share
            else rng.choice(vocab["words"])
            for _ in range(words)
        ]
        text = " ".join(picked)
        return text[0].upper() + text[1:] + "."

    def body(self, key: str, lang: str, lines: int = 18, stop_share: float = 0.3) -> list[str]:
        rng = self.rng(key)
        return [self.sentence(rng, lang, rng.randint(12, 18), stop_share) for _ in range(lines)]

    def add(self, key: str, url: str, lines: list[str], intent: Intent, day: int = 4) -> list[str]:
        self.pages.append(Page(key, url, day, lines, intent))
        return lines


FOOTERS = {
    "en": "Golden News writers keep the river and the garden stories with the readers.",
    "fr": "Les auteurs du blog gardent les histoires de la rivière et du jardin pour vous.",
    "ar": "يحفظ كتاب الأخبار قصص النهر و الحديقة في هذا الموقع من أجل القراء الكرام.",
}
SITES = {"en": "golden-news-en.com", "fr": "golden-blog-fr.org", "ar": "golden-akhbar-ar.net"}
CONTACT = {  # one PII line per site, on its first page
    "en": "Write to the editor at news.desk@golden-news-en.com or call +33 6 12 34 56 78 today.",
    "fr": "Pour nous écrire, utilisez redaction@golden-blog-fr.org et la rivière de la poste.",
    "ar": "للتواصل مع الكاتب في هذا الموقع اتصل على الرقم +216 71 234 567 من الصباح.",
}
CONTACT_PII = {"en": ("email", "phone"), "fr": ("email",), "ar": ("phone",)}


def kept(language: str, pii: tuple[str, ...] = ()) -> Intent:
    """A good page: high tier, in gold."""
    return Intent("high", language, gold=True, pii=pii)


def mixed_sentence(rng: random.Random, words: int) -> str:
    """Words of en, fr and ar in turn: no language wins clearly."""
    picked = [rng.choice(VOCABULARY[("en", "fr", "ar")[i % 3]]["words"]) for i in range(words)]
    return " ".join(picked).capitalize() + "."


def build_pages() -> list[Page]:
    b = Builder()

    # 26 pages on three sites sharing a footer: the domain rule removes it (S2-07).
    for lang, n in (("en", 10), ("fr", 8), ("ar", 8)):
        for i in range(n):
            key = f"site-{lang}-{i}"
            extra = [CONTACT[lang]] if i == 0 else []
            pii = CONTACT_PII[lang] if i == 0 else ()
            lines = ["Home | News | Contact", *b.body(key, lang), *extra, FOOTERS[lang]]
            b.add(key, f"https://www.{SITES[lang]}/{i}", lines, kept(lang, pii))

    # 31 pages alone on their site.
    for lang, n in (("en", 15), ("fr", 8), ("ar", 8)):
        for i in range(n):
            key = f"single-{lang}-{i}"
            b.add(
                key,
                f"http://golden-{lang}-{i}.com/page",
                b.body(key, lang),
                kept(lang),
            )

    # 10 medium pages: short, and only 3 stopwords: kept, scored below 0.8.
    for lang, n in (("en", 4), ("fr", 3), ("ar", 3)):
        for i in range(n):
            key = f"medium-{lang}-{i}"
            lines = b.body(key, lang, lines=8, stop_share=0.0)
            stop = VOCABULARY[lang]["stopwords"]
            lines = [f"{line[:-1]} {stop[j]}." if j < 3 else line for j, line in enumerate(lines)]
            b.add(
                key,
                f"http://golden-medium-{lang}-{i}.com/",
                lines,
                Intent("medium", lang, gold=True),
            )

    # 18 rejected pages, one rule each.
    for lang, n in (("en", 3), ("fr", 2), ("ar", 2)):
        for i in range(n):
            key = f"short-{lang}-{i}"
            b.add(
                key,
                f"http://golden-short-{lang}-{i}.com/",
                b.body(key, lang, lines=3)[:3],
                Intent("rejected", lang, reason="too_few_words"),
            )
    for i in range(3):
        key = f"german-{i}"
        b.add(
            key,
            f"http://golden-de-{i}.de/",
            b.body(key, "de"),
            Intent("rejected", "other", reason="language_not_targeted"),
        )
    for i in range(2):
        rng = b.rng(f"mixed-{i}")
        b.add(
            f"mixed-{i}",
            f"http://golden-mixed-{i}.com/",
            [mixed_sentence(rng, 15) for _ in range(15)],
            Intent("rejected", None, reason="language_low_confidence"),
        )
    lines = b.body("symbols", "en", lines=10)
    lines = [
        " ".join(f"#{w}" if j % 5 == 0 else w for j, w in enumerate(line.split())) for line in lines
    ]
    b.add(
        "symbols",
        "http://golden-symbols.com/",
        lines,
        Intent("rejected", "en", reason="too_many_symbols"),
    )
    lines = b.body("repeated", "en", lines=6)
    b.add(
        "repeated",
        "http://golden-repeated.com/",
        lines + [lines[0]] * 6,
        Intent("rejected", "en", reason="too_many_repeated_lines"),
    )
    lines = b.body("ellipsis", "en", lines=10)
    b.add(
        "ellipsis",
        "http://golden-ellipsis.com/",
        [ln[:-1] + "..." if j < 6 else ln for j, ln in enumerate(lines)],
        Intent("rejected", "en", reason="too_many_ellipsis_lines"),
    )
    words = VOCABULARY["en"]["words"]
    long_words = ["".join(words[k : k + 4]) for k in range(0, 40, 4)]  # ~25 letters each
    lines = [
        " ".join(w if j % 2 else long_words[(j + k) % 10] for j, w in enumerate(ln.split()))
        for k, ln in enumerate(b.body("long-words", "en", lines=12))
    ]
    b.add(
        "long-words",
        "http://golden-long.com/",
        lines,
        Intent("rejected", "en", reason="word_length_out_of_range"),
    )
    b.add(
        "no-stopwords",
        "http://golden-nostop.com/",
        b.body("no-stopwords", "en", stop_share=0.0),
        Intent("rejected", "en", reason="too_few_stopwords"),
    )
    b.add(
        "emptied",
        "http://golden-empty.com/",
        ["Home | About", "Menu", "Login | Sign up", "Top"],
        Intent("rejected", "other", reason="too_few_words"),
    )

    # 13 duplicates: originals fetched first, so the original is the page each group keeps.
    for lang, copies in (("en", 2), ("fr", 1), ("ar", 1)):
        original = b.add(
            f"exact-{lang}-original",
            f"http://golden-origin-{lang}.com/story",
            b.body(f"exact-{lang}", lang),
            kept(lang),
            day=2,
        )
        for c in range(copies):
            lines = original
            if lang == "ar":  # vowel marks added: the same text once arabic_match_key folds it
                lines = [
                    " ".join(
                        w[0] + FATHA + w[1:] if j % 4 == 0 else w for j, w in enumerate(ln.split())
                    )
                    for ln in original
                ]
            b.add(
                f"exact-{lang}-copy-{c}",
                f"http://golden-copy-{lang}-{c}.com/story",
                lines,
                Intent("high", lang, duplicate="exact"),
                day=5 + c,
            )
    for lang in ("en", "fr", "ar"):
        original = b.add(
            f"near-{lang}-original",
            f"http://golden-near-origin-{lang}.com/",
            b.body(f"near-{lang}", lang, lines=20),
            kept(lang),
            day=2,
        )
        changed = list(original)
        for j in (3, 9, 15):  # one content word changed in three lines: Jaccard about 0.9
            first, *rest = changed[j].split(" ", 1)
            changed[j] = " ".join([VOCABULARY[lang]["words"][j], *rest])
        b.add(
            f"near-{lang}-copy",
            f"http://golden-near-copy-{lang}.com/",
            changed,
            Intent("high", lang, duplicate="near"),
            day=6,
        )

    # 2 pages of an excluded site (a subdomain too): kept by silver and dedup, not in gold.
    for i, host in enumerate((EXCLUDED_SITE, f"www.{EXCLUDED_SITE}")):
        b.add(
            f"excluded-{i}",
            f"https://{host}/{i}",
            b.body(f"excluded-{i}", "en"),
            Intent("high", "en"),
        )

    return b.pages


def wet_record(record_id: str, url: str, day: int, body: bytes) -> bytes:
    """One conversion record as its own gzip member, like a real WET file."""
    headers = CRLF.join(
        [
            "WARC/1.0",
            "WARC-Type: conversion",
            f"WARC-Target-URI: {url}",
            f"WARC-Date: 2026-09-{day:02d}T12:00:00Z",
            f"WARC-Record-ID: <urn:uuid:{record_id}>",
            "Content-Type: text/plain",
            f"Content-Length: {len(body)}",
            "",
            "",
        ]
    )
    return gzip.compress(headers.encode("utf-8") + body + (CRLF * 2).encode("ascii"))


def write_bronze(pages: list[Page], folder: Path) -> list[tuple[str, str]]:
    """Two WET files, pages alternating between them, plus one record that is not UTF-8
    (a dead letter). Returns (segment_id, path) as the control table lists them."""
    files: list[tuple[str, str]] = []
    for segment in (0, 1):
        records = [
            wet_record(p.doc_id, p.url, p.day, LF.join(p.lines).encode("utf-8"))
            for p in pages[segment::2]
        ]
        if segment == 1:
            records.append(
                wet_record(
                    str(uuid.uuid5(NAMESPACE, "dead")),
                    "http://golden-dead.com/",
                    4,
                    b"caf" + bytes([0xE9]),
                )
            )
        path = folder / f"segment-{segment}.warc.wet.gz"
        path.write_bytes(b"".join(records))
        files.append((f"{segment:05d}", f"file:{path}"))
    return files


def run_chain(spark: SparkSession, pages: list[Page], tmp: Path) -> dict[str, Any]:
    """Every stage, as the jobs run it, on local folders. Returns the outcome."""
    files = write_bronze(pages, tmp)
    stage1 = build_stage1(spark, files, CRAWL, str(tmp / "stage1"), str(tmp / "dead"))
    docs = spark.read.schema(STAGE1_SCHEMA).parquet(str(tmp / "stage1"))
    root = tmp / "silver_v1"
    silver = run_silver_v1(
        spark, docs, str(root), f"{root}/language=*/quality_tier=*/crawl_id={CRAWL}", CRAWL
    )
    rows = spark.read.schema(SILVER_V1).parquet(str(root)).where(F.col("crawl_id") == CRAWL)
    dedup_out = str(tmp / "dedup" / f"crawl_id={CRAWL}")
    dedup = run_dedup(
        spark, rows.where(F.col("quality_tier") != "rejected"), dedup_out, DedupParams()
    )
    dedup_rows = spark.read.schema(DEDUP_V1).parquet(dedup_out)
    gold, stats = run_gold(spark, rows, dedup_rows, str(tmp / "gold"), CRAWL, (EXCLUDED_SITE,))

    gold_text = {
        r.doc_id: hashlib.sha256(r.text.encode("utf-8")).hexdigest()[:16]
        for r in spark.read.format("delta").load(str(tmp / "gold")).collect()
    }
    duplicates = {r.doc_id: r.duplicate_type for r in dedup_rows.collect()}
    by_id = {p.doc_id: p.key for p in pages}
    outcome: dict[str, Any] = {}
    for r in rows.collect():  # 100 test rows
        outcome[by_id[r.doc_id]] = {
            "language": r.language,
            "tier": r.quality_tier,
            "reasons": r.reject_reasons,
            "pii": r.pii_types,
            "duplicate": duplicates.get(r.doc_id),
            "gold_text": gold_text.get(r.doc_id),
        }
    totals = {
        "documents": stage1["documents"],
        "dead_letters": stage1["dead_letters"],
        "lines_removed_domain": stage1["lines_removed_domain"],
        "tiers": {t: silver.get(f"silver_v1_tier_{t}", 0) for t in ("high", "medium", "rejected")},
        "exact_duplicates": dedup["dedup_exact_duplicates"],
        "near_duplicates": dedup["dedup_near_duplicates"],
        "excluded": gold["gold_documents_excluded"],
        "gold": gold["gold_documents_written"],
        "stats": {f"{s['language']}/{s['quality_tier']}": s["documents"] for s in stats},
    }
    return {"totals": totals, "pages": dict(sorted(outcome.items()))}


@pytest.fixture
def default_file_packing(spark: SparkSession) -> Iterator[None]:
    """build_stage1 changes a session setting (one bronze file per task): put it back."""
    yield
    spark.conf.unset("spark.sql.files.maxPartitionBytes")


@pytest.mark.usefixtures("toy_model", "default_file_packing")
def test_the_golden_pages_land_where_they_should(spark: SparkSession, tmp_path: Path) -> None:
    pages = build_pages()
    assert len(pages) == 100 and len({p.doc_id for p in pages}) == 100
    actual = run_chain(spark, pages, tmp_path)

    # 1. Each page where it was built to land.
    wrong = []
    for page in pages:
        got, want = actual["pages"][page.key], page.intent
        problems = [
            name
            for name, ok in (
                ("tier", got["tier"] == want.tier),
                ("language", want.language in (None, got["language"])),
                ("reason", want.reason is None or want.reason in (got["reasons"] or [])),
                ("duplicate", got["duplicate"] == want.duplicate),
                ("gold", (got["gold_text"] is not None) == want.gold),
                ("pii", sorted(got["pii"] or []) == sorted(want.pii)),
            )
            if not ok
        ]
        if problems:
            wrong.append(f"{page.key}: {problems} -> {got}")
    assert not wrong, "pages that did not land as built:" + LF + LF.join(wrong)

    # 2. Everything exactly as committed: the outcome of every page and the totals.
    if os.environ.get("GOLDEN_WRITE"):
        Path(os.environ["GOLDEN_WRITE"]).write_text(
            json.dumps(actual, indent=1, ensure_ascii=False) + LF, encoding="utf-8"
        )
        pytest.skip(f"expected outcome written to {os.environ['GOLDEN_WRITE']}: review its diff")
    expected = json.loads(EXPECTED.read_text(encoding="utf-8"))
    changed = sorted(k for k in expected["pages"] if expected["pages"][k] != actual["pages"].get(k))
    assert actual["totals"] == expected["totals"], (actual["totals"], expected["totals"])
    assert not changed and actual["pages"].keys() == expected["pages"].keys(), (
        "pages whose outcome changed: " + ", ".join(changed)
    )
