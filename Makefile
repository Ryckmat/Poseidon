.DEFAULT_GOAL := help
PY ?= python
TEST_DB ?= postgresql://poseidon:poseidon@localhost:5432/poseidon_test

.PHONY: help install db-up db-down init-db lint format test test-all run

help: ## Liste les commandes
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

install: ## Installe le projet et les outils de dev
	$(PY) -m pip install -e ".[dev]"
	pre-commit install

db-up: ## Démarre PostgreSQL en local (docker compose)
	docker compose up -d --wait db

db-down: ## Arrête PostgreSQL local
	docker compose down

init-db: ## Crée tables et index
	poseidon init-db

lint: ## Vérifie format et lint
	ruff format --check src tests
	ruff check src tests

format: ## Formate et corrige automatiquement
	ruff format src tests
	ruff check --fix src tests

test: ## Tests unitaires et pipeline (SQLite)
	pytest

test-all: ## Tous les tests, y compris PostgreSQL et dashboard
	POSEIDON_TEST_DATABASE_URL=$(TEST_DB) pytest --cov

run: ## Lance le dashboard
	streamlit run src/poseidon/dashboard/app.py
