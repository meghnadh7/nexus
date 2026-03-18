.PHONY: dev test lint build index eval deploy-staging deploy-prod logs shell help

SHELL := /bin/bash
NAMESPACE := nexus
HELM_CHART := ./infra/helm/nexus
ORCHESTRATOR_POD := $(shell kubectl get pod -n $(NAMESPACE) -l app=nexus-orchestrator -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

dev: ## Start all services with docker-compose
	docker-compose up --build

build: ## Build all Docker images
	docker-compose build

test: ## Run all tests across all services
	@echo "Installing rag-indexer (required by orchestrator)..."
	cd services/rag-indexer && /opt/homebrew/bin/python3.11 -m pip install -e . -q
	@echo "Running orchestrator tests..."
	cd services/orchestrator && /opt/homebrew/bin/python3.11 -m pip install -e ".[dev]" -q && \
		/opt/homebrew/bin/python3.11 -m pytest tests/ -v --tb=short -m "not integration"
	@echo "All tests passed."

test-integration: ## Run integration tests (requires real API keys)
	cd services/orchestrator && pytest tests/ -v --tb=short -m integration

test-docker: ## Run tests inside Docker (isolated)
	docker-compose -f docker-compose.test.yml up --build --abort-on-container-exit --exit-code-from test-runner

lint: ## Run ruff and mypy for all services
	@for svc in orchestrator mcp-github mcp-slack mcp-jira mcp-datadog rag-indexer; do \
		echo "Linting services/$$svc..."; \
		cd services/$$svc && ruff check src/ && mypy src/ --ignore-missing-imports; \
		cd ../..; \
	done

format: ## Auto-format all services with ruff
	@for svc in orchestrator mcp-github mcp-slack mcp-jira mcp-datadog rag-indexer; do \
		echo "Formatting services/$$svc..."; \
		cd services/$$svc && ruff format src/; \
		cd ../..; \
	done

index: ## Run RAG indexer against local repository
	@echo "Running RAG indexer against $(CURDIR)..."
	docker-compose --profile indexer run --rm \
		-e REPO_PATH=/repo \
		-v $(CURDIR):/repo:ro \
		rag-indexer

eval: ## Run LangSmith evaluation harness
	@echo "Running LangSmith evaluations..."
	pip install langsmith -q
	python evals/run_evals.py

deploy-staging: ## Deploy to staging with Helm
	helm upgrade --install nexus $(HELM_CHART) \
		--namespace $(NAMESPACE) \
		--create-namespace \
		--values $(HELM_CHART)/values.yaml \
		--set global.imageTag=$(shell git rev-parse --short HEAD) \
		--wait --timeout 5m
	kubectl rollout status deployment/nexus-orchestrator -n $(NAMESPACE)
	@echo "✅ Deployed to staging successfully."

deploy-prod: ## Deploy to production with Helm (requires confirmation)
	@echo "⚠️  You are about to deploy to PRODUCTION. Are you sure? [y/N]" && read ans && [ $${ans:-N} = y ]
	helm upgrade --install nexus $(HELM_CHART) \
		--namespace $(NAMESPACE) \
		--create-namespace \
		--values $(HELM_CHART)/values.yaml \
		--values $(HELM_CHART)/values-prod.yaml \
		--set global.imageTag=$(shell git rev-parse --short HEAD) \
		--wait --timeout 10m
	kubectl rollout status deployment/nexus-orchestrator -n $(NAMESPACE)
	@echo "✅ Deployed to production successfully."

logs: ## Tail orchestrator logs
	kubectl logs -f -n $(NAMESPACE) -l app=nexus-orchestrator --max-log-requests=5

shell: ## Open a shell in the orchestrator pod
	kubectl exec -it -n $(NAMESPACE) $(ORCHESTRATOR_POD) -- /bin/bash

db-migrate: ## Run Alembic migrations
	cd services/orchestrator && alembic upgrade head

db-rollback: ## Roll back last Alembic migration
	cd services/orchestrator && alembic downgrade -1

clean: ## Remove all Docker containers and volumes
	docker-compose down -v --remove-orphans
