-- Story 2.4 (DECISIONS S2-10): dev runs (5 segments) must never feed the gate baselines
-- of Sprint 3 (T11, T13). A run is marked here, and the baseline queries exclude it.
-- Existing runs were all full runs, hence the default.
ALTER TABLE pipeline_runs ADD COLUMN dev_mode BOOLEAN NOT NULL DEFAULT false;
