# Thin wrappers around docker compose and uv: no shell-specific syntax,
# so the same targets work from PowerShell, cmd or Git Bash.

.PHONY: up down logs build test lint clean-silver clean-gold

# Start every service in the background; --wait blocks until healthchecks pass.
up:
	docker compose up -d --wait

# Stop and remove containers. Named volumes (MinIO, Postgres data) are kept.
down:
	docker compose down

# Follow logs of all services, or one: make logs SERVICE=airflow-scheduler
logs:
	docker compose logs -f --tail=200 $(SERVICE)

# Build the corpus wheel into dist/.
build:
	uv build --wheel

test:
	$(error make test: not implemented yet (Story 0.4, runs pytest in a Linux container))

lint:
	$(error make lint: not implemented yet (Story 0.4, ruff + mypy))

# clean-* delete MinIO data and never touch corpus-bronze (invariant 1).
# Deleting data needs an explicit: make clean-silver CONFIRM=yes
# clean-gold empties corpus-gold, which also holds signature_store and dup_clusters:
# they describe what gold contains, so they go with it.
clean-silver:
ifneq ($(CONFIRM),yes)
	$(error clean-silver deletes all of corpus-silver. Rerun with CONFIRM=yes)
endif
	docker compose run --rm --entrypoint /bin/sh minio-init /minio-clean.sh corpus-silver

clean-gold:
ifneq ($(CONFIRM),yes)
	$(error clean-gold deletes all of corpus-gold (incl. signature_store, dup_clusters). Rerun with CONFIRM=yes)
endif
	docker compose run --rm --entrypoint /bin/sh minio-init /minio-clean.sh corpus-gold
