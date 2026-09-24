-- Story 1.6: make a run's numbers permanent. Every stage emits its own metrics in
-- the sprint that builds it, never retroactively (CLAUDE.md plan rules).

-- Long format on purpose: Sprint 2 adds "documents parsed", Sprint 4 "duplicate
-- rates", and each new metric is a new ROW, never a migration. Reading it means
-- pivoting, which is dbt's job in Sprint 8.
CREATE TABLE run_metrics (
    run_id      TEXT             NOT NULL,   -- Airflow's dag_run.run_id (T8)
    crawl_id    TEXT             NOT NULL,
    stage       TEXT             NOT NULL,   -- bronze (S1), silver (S2-S3), dedup, gold
    metric      TEXT             NOT NULL,   -- e.g. segments_complete, bytes_downloaded
    value       DOUBLE PRECISION NOT NULL,
    recorded_at TIMESTAMPTZ      NOT NULL DEFAULT now(),
    -- Without this key a retried task would write its rows twice and every later
    -- average would be wrong (T8: unique run_metrics, written with an upsert).
    PRIMARY KEY (run_id, stage, metric)
);

-- "This metric over time for this crawl" is the dashboard query (S8).
CREATE INDEX run_metrics_crawl_metric_idx ON run_metrics (crawl_id, stage, metric, recorded_at);

-- One row per run: what the run WAS, rather than what it measured (T8). The seed and
-- the segment count are what make a run reproducible months later.
CREATE TABLE pipeline_runs (
    run_id        TEXT        PRIMARY KEY,
    crawl_id      TEXT        NOT NULL,
    started_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at      TIMESTAMPTZ,
    status        TEXT        NOT NULL,   -- running|success|failed
    segment_count INT,
    sample_seed   INT,
    CONSTRAINT pipeline_runs_status_valid CHECK (status IN ('running', 'success', 'failed'))
);

CREATE INDEX pipeline_runs_crawl_idx ON pipeline_runs (crawl_id, started_at);

-- Established at download time (the ETag is an MD5 for single-part objects, S1-04)
-- but until now not stored. NULL means "not established", false means "not comparable".
ALTER TABLE segments ADD COLUMN checksum_verified BOOLEAN;
