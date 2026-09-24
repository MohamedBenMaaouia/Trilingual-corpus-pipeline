"""Numbered .sql files applied in order by corpus.db.apply_migrations().

They live inside the package so they travel with the wheel: the Airflow container
mounts only src/, and on Databricks (S10) only the installed wheel exists.
"""
