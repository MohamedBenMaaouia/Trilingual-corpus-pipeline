"""Stage 3e-3f: clusters and the page each one keeps (Story 4.4, DECISIONS S4-05).

Connected components by label propagation on DataFrames (plan 4.4.1: no GraphFrames
dependency), then one representative per cluster (plan 4.4.4).
"""

from pyspark import StorageLevel
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from corpus.dedup.exact import keep_order

EXACT, NEAR = "exact", "near"  # duplicate_type (schemas.dedup_v1)


def connected_components(edges: DataFrame, max_rounds: int) -> tuple[DataFrame, int, bool]:
    """(doc_id, component) for every page in an edge; component = the smallest doc_id of
    the page's connected group. Returns (labels, rounds, converged).

    Every page starts with the smallest doc_id among itself and its neighbours; each
    round, it takes the smallest label among its own and its neighbours' labels. When a
    round changes nothing, every page carries its group's smallest doc_id. A round costs
    one join and one aggregation; the rounds needed grow with the longest chain of pages
    (plan 4.4.1). Each round's labels are local-checkpointed (kept on the executors,
    lineage cut), or the query plan would grow with every round (trap S4).

    If `max_rounds` is reached first, a group may still carry several labels: it is
    split into smaller clusters (fewer pages removed, never unrelated pages merged), and
    the caller logs it (converged is False).
    """
    links = (
        edges.select(F.col("doc_a").alias("src"), F.col("doc_b").alias("dst"))
        .union(edges.select(F.col("doc_b").alias("src"), F.col("doc_a").alias("dst")))
        .persist(StorageLevel.DISK_ONLY)  # read by every round
    )
    labels = (
        links.groupBy(F.col("src").alias("doc_id"))
        .agg(F.min("dst").alias("smallest_neighbour"))
        .select("doc_id", F.least("doc_id", "smallest_neighbour").alias("component"))
        .localCheckpoint()
    )
    rounds = 1
    converged = False
    while rounds < max_rounds:
        proposed = (
            links.join(labels, links["src"] == labels["doc_id"])
            .groupBy(F.col("dst").alias("doc_id"))
            .agg(F.min("component").alias("proposed"))
        )
        updated = (
            labels.join(proposed, "doc_id")
            .select(
                "doc_id",
                F.least("component", "proposed").alias("component"),
                (F.col("proposed") < F.col("component")).alias("changed"),
            )
            .localCheckpoint()
        )
        rounds += 1
        changed = updated.where("changed").count()
        labels = updated.drop("changed")
        if changed == 0:
            converged = True
            break
    links.unpersist()
    return labels, rounds, converged


def assign_clusters(docs: DataFrame, components: DataFrame) -> DataFrame:
    """One row per page: doc_id, language, cluster_id, cluster_size, duplicate_type.

    `docs`: doc_id, language, quality_score, fetch_date, exact_rep (exact.py).
    `components`: (doc_id, component) for the exact representatives with a verified
    near duplicate (connected_components).

    A page's group is its exact representative's component, or that representative
    itself when it has no near duplicate. The group keeps one page by keep_order (plan
    4.4.4): its doc_id is the cluster_id, so gold keeps the rows where doc_id equals
    cluster_id. The kept page is always an exact representative: within an exact
    group, the representative is the best page by the same order.

    duplicate_type: null for the kept page; "exact" for a page with the same text as
    another (stage 3a); "near" for an exact representative that lost to a near duplicate.
    """
    labelled = docs.join(
        components.select(F.col("doc_id").alias("exact_rep"), "component"), "exact_rep", "left"
    ).withColumn("group_id", F.coalesce("component", "exact_rep"))
    clustered = labelled.select(
        "doc_id",
        "language",
        "exact_rep",
        F.first("doc_id")
        .over(Window.partitionBy("group_id").orderBy(*keep_order()))
        .alias("cluster_id"),
        F.count("*").over(Window.partitionBy("group_id")).cast("int").alias("cluster_size"),
    )
    duplicate_type = (
        F.when(F.col("doc_id") == F.col("cluster_id"), F.lit(None).cast("string"))
        .when(F.col("doc_id") != F.col("exact_rep"), F.lit(EXACT))
        .otherwise(F.lit(NEAR))
    )
    return clustered.select(
        "doc_id", "language", "cluster_id", "cluster_size", duplicate_type.alias("duplicate_type")
    )
