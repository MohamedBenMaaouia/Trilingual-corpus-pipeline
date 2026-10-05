-- Story 6.4.2 (C21, DECISIONS S6-07): what each run put in gold, one row per run x
-- language x quality tier. Telemetry, like run_metrics: Postgres locally, a Delta table
-- in the meta schema on Azure (Sprint 11). Written by the gold job; read by dbt (S8).
CREATE TABLE corpus_stats (
    run_id           TEXT             NOT NULL,
    crawl_id         TEXT             NOT NULL,
    language         TEXT             NOT NULL,   -- en | fr | ar
    quality_tier     TEXT             NOT NULL,   -- high | medium
    documents        BIGINT           NOT NULL,
    chars_total      BIGINT           NOT NULL,
    words_total      BIGINT           NOT NULL,
    mean_chars       DOUBLE PRECISION NOT NULL,   -- mean length, in characters
    first_fetch      TIMESTAMPTZ      NOT NULL,   -- the date range of the pages
    last_fetch       TIMESTAMPTZ      NOT NULL,
    pipeline_version TEXT             NOT NULL,
    recorded_at      TIMESTAMPTZ      NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, language, quality_tier)
);

-- "The corpus over time, crawl by crawl" is the dashboard query (S8).
CREATE INDEX corpus_stats_crawl_idx ON corpus_stats (crawl_id, recorded_at);
