-- Story 1.5: every gate records its verdict from Sprint 1 on (C12 moves this out of S8).
-- Airflow shows that a task went red; this says which check failed and by how much.
CREATE TABLE gate_results (
    -- Airflow's dag_run.run_id: ties the verdict to one exact DAG run (T8).
    run_id         TEXT        NOT NULL,
    crawl_id       TEXT        NOT NULL,
    gate_name      TEXT        NOT NULL,   -- gate_a (S1), gate_b (S3), gate_c (S5)
    passed         BOOLEAN     NOT NULL,
    failure_detail TEXT,                   -- what the checks reported when passed = false
    checked_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- A retried task replaces its row instead of adding one (invariant 7). The
    -- per-attempt history lives in Airflow; here the last verdict is the verdict.
    PRIMARY KEY (run_id, gate_name)
);

-- "How did gate_a do across this crawl's runs?" is the dashboard query (S8).
CREATE INDEX gate_results_crawl_gate_idx ON gate_results (crawl_id, gate_name, checked_at);
