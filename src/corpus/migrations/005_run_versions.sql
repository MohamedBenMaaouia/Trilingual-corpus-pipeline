-- Story 3.4 (D8, DECISIONS S3-01): which logic and which contract produced a run's
-- data. Left out by S1-07 until a stage used them; silver_v1 is the first. NULL for
-- runs before Sprint 3 and for runs that wrote no silver_v1 data.
ALTER TABLE pipeline_runs ADD COLUMN pipeline_version TEXT;
ALTER TABLE pipeline_runs ADD COLUMN schema_version TEXT;
