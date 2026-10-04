"""The pipeline's logic version (D8, DECISIONS S3-01).

Stamped into every silver row and recorded on the run. Separate from the package
version: bump it only when transform logic or a threshold changes, because a bump
means a documented backfill of every retained crawl (invariant 9). A change to
tests, docs or the DAG alone does not bump it.

History:
1  Sprint 3: first silver_v1 (language ID, quality, PII; DECISIONS S3-04 to S3-09).
"""

PIPELINE_VERSION = "1"
