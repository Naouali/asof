# QuantLab -- single entrypoint for every common task.
#
# Nothing here requires a Python toolchain on the host. Docker is the only
# prerequisite.

SHELL := /bin/bash
.DEFAULT_GOAL := help
.PHONY: help base build up down restart ps logs doctor init ingest ingest-daily \
        test lint format typecheck check shell catalogue status fetchers \
        ui ui-dev ui-check serve lock clean clean-data verify-multiarch dev-image

COMPOSE      ?= docker compose
BASE_IMAGE   ?= quantlab-base:local
DEV_IMAGE    ?= quantlab-dev:local
UID          ?= $(shell id -u)
GID          ?= $(shell id -g)
BUILD_ARGS    = --build-arg UID=$(UID) --build-arg GID=$(GID)
# Run one-off commands in a throwaway container with the repo mounted, so tests
# and linters see the working tree rather than a stale image layer.
DEV_RUN       = docker run --rm -t -v "$(PWD)/src:/app/src:ro" -v "$(PWD)/tests:/app/tests:ro" \
                -v "$(PWD)/configs:/app/configs:ro" -v "$(PWD)/scripts:/app/scripts:ro" \
                -v "$(PWD)/docs:/app/docs:ro" -v "$(PWD)/README.md:/app/README.md:ro" \
                -v "$(PWD)/Makefile:/app/Makefile:ro" -v "$(PWD)/docker:/app/docker:ro" \
                -v "$(PWD)/docker-compose.yml:/app/docker-compose.yml:ro" \
                -v "$(PWD)/docker-compose.override.yml.example:/app/docker-compose.override.yml.example:ro" \
                -v "$(PWD)/uv.lock:/app/uv.lock:ro" -v "$(PWD)/.gitignore:/app/.gitignore:ro" \
                -v "$(PWD)/.env.example:/app/.env.example:ro" \
                -v "$(PWD)/pyproject.toml:/app/pyproject.toml:ro" -w /app $(DEV_IMAGE)

help:  ## Show this help
	@echo "QuantLab -- make targets"
	@echo
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "First run:  cp .env.example .env && make up"

# ----------------------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------------------
base:  ## Build the shared base image (every service image builds FROM it)
	docker build -f docker/Dockerfile.base --target runtime $(BUILD_ARGS) -t $(BASE_IMAGE) .

dev-image:  ## Build the dev image (test + lint tooling)
	docker build -f docker/Dockerfile.base --target dev $(BUILD_ARGS) -t $(DEV_IMAGE) .

build: base  ## Build every service image
	$(COMPOSE) build

# ----------------------------------------------------------------------------------
# Stack lifecycle
# ----------------------------------------------------------------------------------
up: build  ## Start the full stack and wait for every service to report healthy
	$(COMPOSE) up -d --wait
	@echo
	@$(COMPOSE) ps
	@echo
	@echo "app  http://127.0.0.1:$${QUANTLAB_WEB_PORT:-8080}   (no login; this machine only)"
	@echo
	@$(MAKE) --no-print-directory doctor

down:  ## Stop the stack (the data volume survives)
	$(COMPOSE) down --remove-orphans

restart: down up  ## Restart the stack

ps:  ## Show service status
	$(COMPOSE) ps

logs:  ## Tail logs from every service (SERVICE=worker to narrow)
	$(COMPOSE) logs -f --tail=100 $(SERVICE)

# ----------------------------------------------------------------------------------
# Operations
# ----------------------------------------------------------------------------------
doctor:  ## Report what this installation can and cannot do
	$(COMPOSE) exec -T worker quantlab doctor

init:  ## Create the data lake directory structure
	$(COMPOSE) exec -T worker quantlab init

ingest:  ## Full historical data pull
	$(COMPOSE) exec -T worker quantlab data ingest

ingest-daily:  ## Incremental data update
	$(COMPOSE) exec -T worker quantlab data ingest --incremental

catalogue:  ## List data sources, their availability and their caveats
	$(COMPOSE) exec -T worker quantlab data catalogue

status:  ## Show what the lake holds and how stale it is
	$(COMPOSE) exec -T worker quantlab data status

fetchers:  ## List every fetcher and whether it can run right now
	$(COMPOSE) exec -T worker quantlab data fetchers

shell:  ## Open a shell in the worker container
	$(COMPOSE) exec worker /bin/bash

# ----------------------------------------------------------------------------------
# The web app, without Docker (needs Node and the Python environment on the host)
# ----------------------------------------------------------------------------------
ui:  ## Build the interface into ui/dist
	cd ui && npm ci && npm run build

serve: ## Serve the app on http://127.0.0.1:8080 (run `make ui` first)
	.venv/bin/quantlab serve --port $(or $(PORT),8080)

ui-dev:  ## Interface with hot reload on :5173, proxying /api to `make serve`
	cd ui && npm run dev

ui-check:  ## Type-check and test the interface
	cd ui && npm run typecheck && npm test

# ----------------------------------------------------------------------------------
# Quality
# ----------------------------------------------------------------------------------
test: dev-image  ## Run the test suite (excludes the Docker integration tests)
	$(DEV_RUN) pytest

lint: dev-image  ## Lint
	$(DEV_RUN) ruff check src tests scripts
	$(DEV_RUN) ruff format --check src tests scripts

format: dev-image  ## Auto-format (writes to the working tree)
	docker run --rm -t -v "$(PWD)/src:/app/src" -v "$(PWD)/tests:/app/tests" \
	  -v "$(PWD)/scripts:/app/scripts" -v "$(PWD)/pyproject.toml:/app/pyproject.toml:ro" \
	  -w /app $(DEV_IMAGE) \
	  bash -lc "ruff check --fix src tests scripts && ruff format src tests scripts"

typecheck: dev-image  ## Type-check
	$(DEV_RUN) mypy

check: lint typecheck test  ## Everything CI runs

lock: ## Re-resolve dependencies and rewrite uv.lock (run after editing pyproject.toml)
	docker run --rm -v "$(PWD):/w" -w /w ghcr.io/astral-sh/uv:0.5.14 uv lock

verify-multiarch:  ## Confirm the base image builds for both architectures
	docker buildx build --platform linux/amd64,linux/arm64 \
	  -f docker/Dockerfile.base --target runtime $(BUILD_ARGS) -t $(BASE_IMAGE)-multiarch .

# ----------------------------------------------------------------------------------
# Cleanup
# ----------------------------------------------------------------------------------
clean:  ## Remove build caches and images (keeps the data volume)
	$(COMPOSE) down --remove-orphans --rmi local || true
	docker image rm -f $(BASE_IMAGE) $(DEV_IMAGE) 2>/dev/null || true

clean-data:  ## Wipe the data lake. Irreversible; asks first.
	@echo "This permanently deletes the quantlab-data volume: every ingested dataset,"
	@echo "every snapshot, and the options chain history, which CANNOT be re-downloaded"
	@echo "because no free source sells historical options chains."
	@read -r -p "Type 'delete' to confirm: " reply; \
	  if [ "$$reply" = "delete" ]; then \
	    $(COMPOSE) down --remove-orphans; \
	    docker volume rm quantlab-data || true; \
	    echo "data volume removed"; \
	  else \
	    echo "aborted; nothing was deleted"; \
	  fi
