"""dedup.clusters: connected components and each cluster's kept page (Story 4.4, S4-05)."""

from datetime import UTC, datetime

from pyspark.sql import DataFrame, SparkSession

from corpus.dedup.clusters import assign_clusters, connected_components

DAY = [datetime(2026, 9, d, tzinfo=UTC) for d in range(1, 5)]


def edges(spark: SparkSession, pairs: list[tuple[str, str]]) -> DataFrame:
    return spark.createDataFrame(pairs, "doc_a string, doc_b string")


def test_components_carry_their_smallest_doc_id(spark: SparkSession) -> None:
    # A chain e-d-c-b-a (the smallest at the far end) and a separate pair x-y.
    graph = edges(spark, [("d", "e"), ("c", "d"), ("b", "c"), ("a", "b"), ("x", "y")])
    labels, rounds, converged = connected_components(graph, max_rounds=30)
    assert converged
    assert {r.doc_id: r.component for r in labels.collect()} == {
        "a": "a",
        "b": "a",
        "c": "a",
        "d": "a",
        "e": "a",
        "x": "x",
        "y": "x",
    }
    # Round 1 looks one step away; "e" is 4 steps from "a": 3 more rounds, then one
    # round that changes nothing.
    assert rounds == 5


def test_the_round_cap_stops_and_says_so(spark: SparkSession) -> None:
    graph = edges(spark, [("d", "e"), ("c", "d"), ("b", "c"), ("a", "b")])
    labels, rounds, converged = connected_components(graph, max_rounds=2)
    assert (rounds, converged) == (2, False)
    # Not converged: "e" has not heard of "a" yet; it is never merged with a stranger.
    assert {r.doc_id: r.component for r in labels.collect()}["e"] == "c"


def test_each_cluster_keeps_one_page_and_names_the_others(spark: SparkSession) -> None:
    docs = spark.createDataFrame(
        [
            # a1 and a2 have the same text; a2 is the earlier copy, so it represents them.
            ("a1", "en", 0.9, DAY[1], "a2"),
            ("a2", "en", 0.9, DAY[0], "a2"),
            # a3 is a near duplicate of a2 with a higher score: it is the one kept.
            ("a3", "en", 0.95, DAY[2], "a3"),
            ("b1", "fr", 0.4, DAY[0], "b1"),  # no duplicate
        ],
        "doc_id string, language string, quality_score double, fetch_date timestamp, "
        "exact_rep string",
    )
    components = spark.createDataFrame(
        [("a2", "a2"), ("a3", "a2")], "doc_id string, component string"
    )
    result = {r.doc_id: r for r in assign_clusters(docs, components).collect()}
    assert {d: (r.cluster_id, r.cluster_size, r.duplicate_type) for d, r in result.items()} == {
        "a1": ("a3", 3, "exact"),
        "a2": ("a3", 3, "near"),
        "a3": ("a3", 3, None),
        "b1": ("b1", 1, None),
    }
    assert result["b1"].language == "fr"
