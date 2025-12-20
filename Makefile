.PHONY: install test lint format docker-build docker-up docker-down clean help

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	pip install -r requirements.txt
	pre-commit install

test:  ## Run tests
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint:  ## Run linting
	flake8 src/ tests/ --max-line-length=120 --extend-ignore=E203,E266,E501

format:  ## Format code with black
	black src/ tests/ app.py --line-length=120

run-app:  ## Run Flask application
	python app.py

run-mlflow:  ## Start MLflow UI
	mlflow ui --host 0.0.0.0 --port 5000

docker-build:  ## Build Docker image
	docker build -t fakenews-detection-ml:latest .

docker-up:  ## Start all services with Docker Compose
	docker-compose up -d

docker-down:  ## Stop all services
	docker-compose down

docker-logs:  ## View Docker logs
	docker-compose logs -f

dvc-repro:  ## Run DVC pipeline
	dvc repro

dvc-dag:  ## Show DVC pipeline DAG
	dvc dag

clean:  ## Clean up generated files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache htmlcov .coverage

setup:  ## Initial project setup
	python3 -m venv venv
	@echo "Virtual environment created. Activate with: source venv/bin/activate"
	@echo "Then run: make install"

check:  ## Run all checks (lint, format check, tests)
	black --check src/ tests/ app.py
	flake8 src/ tests/ --max-line-length=120
	pytest tests/ -v

ci:  ## Run CI pipeline locally
	make lint
	make test
	make docker-build
