-- Runs once, when the postgres volume is empty (docker-entrypoint-initdb.d).
-- airflow: Airflow's own metadata. corpus: the pipeline's control plane and metrics (S1+).
CREATE DATABASE airflow;
CREATE DATABASE corpus;
