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
clean-silver:
	$(error make clean-silver: not implemented yet (Story 0.2, needs the MinIO client service))

clean-gold:
	$(error make clean-gold: not implemented yet (Story 0.2, needs the MinIO client service))
