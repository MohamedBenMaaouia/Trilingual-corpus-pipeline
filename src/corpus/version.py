"""The pipeline's logic version (D8, DECISIONS S3-01).

Stamped into every silver row and recorded on the run. Separate from the package
version: bump it only when transform logic or a threshold changes, because a bump
means a documented backfill of every retained crawl (invariant 9). A change to
tests, docs or the DAG alone does not bump it.

History:
1  Sprint 3: first silver_v1 (language ID, quality, PII; DECISIONS S3-04 to S3-09).
   Sprint 4: first dedup (exact + MinHash 128, 5-grams, seed 42, LSH 16 x 8, pairs
   checked at t; DECISIONS S4-02 to S4-07). Added, not changed: nothing downstream
   existed, so no backfill (S4-06). Any later change to these bumps the version.
   Sprint 6: first gold (gold_v1, DECISIONS S6-02 to S6-08). Added, not changed: no
   backfill. The exclusion list is data, not logic: changing it does not bump (S6-05).
"""

PIPELINE_VERSION = "1"
