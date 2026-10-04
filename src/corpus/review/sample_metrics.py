"""The three Sprint 3 quality numbers from the user's labels (Story 3.6.4, DECISIONS S3-10).

Each sample is stratified, so every estimate weights a group's labelled share by the
group's size in the crawl (the record's group_sizes). Only labelled items count; each
number comes with its sample size, and single-group shares with a 95% Wilson interval.

    python -m corpus.review.sample_metrics lid --record ... --labels ...
    python -m corpus.review.sample_metrics pii --record ... --labels ...
    python -m corpus.review.sample_metrics rejected --record ... --labels ...
"""

import argparse
import csv
import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

TARGETS = ("en", "fr", "ar")


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """The 95% Wilson score interval of successes/n (None when n is 0). Better than
    "p +- 1.96 sqrt(p(1-p)/n)" for small n and shares near 0 or 1."""
    if n == 0:
        return None
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)


def read_labels(path: Path) -> dict[str, str]:
    """id -> label from an exported CSV (header doc_id,label or card_id,label)."""
    with path.open(encoding="utf-8", newline="") as handle:
        return {row[0]: row[1] for row in list(csv.reader(handle))[1:] if len(row) >= 2}


def _weighted(
    items: Sequence[Mapping[str, Any]],
    labels: Mapping[str, str],
    sizes: Mapping[str, int],
    id_key: str,
    hit: Callable[[Mapping[str, Any], str], bool],
    strata: Sequence[str] | None = None,
) -> tuple[float, int]:
    """sum over groups of size x (labelled share where `hit`) -> (estimated count, n used)."""
    total, used = 0.0, 0
    for stratum in strata or sorted(sizes):
        labelled = [i for i in items if i["stratum"] == stratum and i[id_key] in labels]
        if labelled:
            share = sum(hit(i, labels[i[id_key]]) for i in labelled) / len(labelled)
            total += sizes[stratum] * share
            used += len(labelled)
    return total, used


def _observed_weight(
    items: Sequence[Mapping[str, Any]],
    labels: Mapping[str, str],
    sizes: Mapping[str, int],
    id_key: str,
) -> int:
    """The total size of the groups that have at least one labelled item."""
    observed = {i["stratum"] for i in items if i[id_key] in labels}
    return sum(size for stratum, size in sizes.items() if stratum in observed)


def _is_label(language: str) -> Callable[[Mapping[str, Any], str], bool]:
    def hit(_: Mapping[str, Any], label: str) -> bool:
        return label == language

    return hit


def _kept_and_is(language: str) -> Callable[[Mapping[str, Any], str], bool]:
    def hit(page: Mapping[str, Any], label: str) -> bool:
        return label == language and page["stratum"] == f"{language}_high"

    return hit


def _fasttext_right(page: Mapping[str, Any], label: str) -> bool:
    """Did fastText's top label match the user's? ar-dialect matches fastText's Egyptian
    "arz"; "other" matches any non-target label; "mixed" never matches."""
    detected = page["language_detected"]
    if label in TARGETS:
        return bool(detected == label)
    if label == "ar-dialect":
        return bool(detected == "arz")
    return label == "other" and detected not in (*TARGETS, "arz")


def lid_report(record: Mapping[str, Any], labels: Mapping[str, str]) -> dict[str, Any]:
    """Per target language: precision of fastText's kept pages (the _high group), recall
    over the pages either detector found (recall cannot see pages both missed, S3-06),
    and overall accuracy; plus accuracy by confidence band (unweighted)."""
    pages, sizes = record["pages"], record["group_sizes"]
    labelled = [p for p in pages if p["doc_id"] in labels]
    report: dict[str, Any] = {
        "labelled": len(labelled),
        "mixed": sum(labels[p["doc_id"]] == "mixed" for p in labelled),
        "per_language": {},
    }
    for language in TARGETS:
        high = [p for p in labelled if p["stratum"] == f"{language}_high"]
        correct = sum(labels[p["doc_id"]] == language for p in high)
        found, _ = _weighted(pages, labels, sizes, "doc_id", _kept_and_is(language))
        truth, used = _weighted(pages, labels, sizes, "doc_id", _is_label(language))
        report["per_language"][language] = {
            "precision": round(correct / len(high), 4) if high else None,
            "precision_n": len(high),
            "precision_95": wilson(correct, len(high)),
            "recall": round(found / truth, 4) if truth else None,
            "recall_n": used,
        }

    agree, used = _weighted(pages, labels, sizes, "doc_id", _fasttext_right)
    weight = _observed_weight(pages, labels, sizes, "doc_id")
    report["accuracy"] = round(agree / weight, 4) if weight else None
    report["accuracy_n"] = used

    report["by_confidence"] = []
    for low, high_end in [(0.0, 0.5), (0.5, 0.65), (0.65, 0.8), (0.8, 1.01)]:
        inside = [
            p
            for p in labelled
            if p["language_detected"] in TARGETS and low <= p["language_conf"] < high_end
        ]
        hits = sum(labels[p["doc_id"]] == p["language_detected"] for p in inside)
        report["by_confidence"].append(
            {
                "band": [low, min(high_end, 1.0)],
                "n": len(inside),
                "correct": hits,
                "share_95": wilson(hits, len(inside)),
            }
        )
    return report


def pii_report(record: Mapping[str, Any], labels: Mapping[str, str]) -> dict[str, Any]:
    """Redaction precision (share of caught candidates that are PII) and recall (caught
    PII over all PII among candidates, weighted by the two groups' sizes)."""
    items, sizes = record["candidates"], record["group_sizes"]

    def is_pii(_: Mapping[str, Any], label: str) -> bool:
        return label in ("email", "phone")

    caught = [c for c in items if c["stratum"] == "pii_caught" and c["candidate_id"] in labels]
    true_caught = sum(is_pii(c, labels[c["candidate_id"]]) for c in caught)
    found, _ = _weighted(items, labels, sizes, "candidate_id", is_pii, ["pii_caught"])
    total, used = _weighted(items, labels, sizes, "candidate_id", is_pii)
    return {
        "labelled": sum(c["candidate_id"] in labels for c in items),
        "precision": round(true_caught / len(caught), 4) if caught else None,
        "precision_n": len(caught),
        "precision_95": wilson(true_caught, len(caught)),
        "recall": round(found / total, 4) if total else None,
        "recall_n": used,
    }


def rejected_report(record: Mapping[str, Any], labels: Mapping[str, str]) -> dict[str, Any]:
    """Filter precision: the share of rejected pages that were rightly rejected (junk),
    per language and weighted overall; "unsure" answers are left out and counted."""
    pages, sizes = record["pages"], record["group_sizes"]
    sure = {k: v for k, v in labels.items() if v in ("good", "junk")}
    report: dict[str, Any] = {
        "labelled": sum(p["doc_id"] in labels for p in pages),
        "unsure": sum(labels.get(p["doc_id"]) == "unsure" for p in pages),
        "per_language": {},
    }
    for language in TARGETS:
        group = [p for p in pages if p["stratum"] == f"rejected_{language}" and p["doc_id"] in sure]
        junk = sum(sure[p["doc_id"]] == "junk" for p in group)
        report["per_language"][language] = {
            "precision": round(junk / len(group), 4) if group else None,
            "n": len(group),
            "precision_95": wilson(junk, len(group)),
        }
    rightly, used = _weighted(pages, sure, sizes, "doc_id", _is_label("junk"))
    weight = _observed_weight(pages, sure, sizes, "doc_id")
    report["precision"] = round(rightly / weight, 4) if weight else None
    report["precision_n"] = used
    return report


REPORTS: dict[str, Callable[[Mapping[str, Any], Mapping[str, str]], dict[str, Any]]] = {
    "lid": lid_report,
    "pii": pii_report,
    "rejected": rejected_report,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample", choices=sorted(REPORTS))
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    args = parser.parse_args()
    record = json.loads(args.record.read_text(encoding="utf-8"))
    print(json.dumps(REPORTS[args.sample](record, read_labels(args.labels)), indent=1))


if __name__ == "__main__":
    main()
