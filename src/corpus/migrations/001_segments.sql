-- Story 1.2: the control plane. One row per WET file this run intends to have.
-- It is the pipeline's memory: a restart reads it to know what is already done.
CREATE TABLE segments (
    crawl_id     TEXT        NOT NULL,
    -- 5-digit position of the file in the crawl's manifest (T6). TEXT keeps the
    -- leading zeros; the number in a WET filename is NOT unique within a crawl.
    segment_id   TEXT        NOT NULL,
    source_url   TEXT        NOT NULL,
    status       TEXT        NOT NULL,
    attempts     INT         NOT NULL DEFAULT 0,
    bytes        BIGINT,
    etag         TEXT,       -- recorded for reference only, not a checksum we verify (C2)
    final_key    TEXT,
    error        TEXT,
    -- When a downloader took this row (T8). A killed process leaves status
    -- 'downloading' forever; the age of this lease is what lets a restart reclaim it.
    claimed_at   TIMESTAMPTZ,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ,
    PRIMARY KEY (crawl_id, segment_id),
    -- A typo like 'compleet' would silently become a state nothing matches.
    CONSTRAINT segments_status_valid
        CHECK (status IN ('pending', 'downloading', 'complete', 'failed'))
);

-- "Give me this crawl's pending segments" is the downloader's hot query.
CREATE INDEX segments_crawl_status_idx ON segments (crawl_id, status);
