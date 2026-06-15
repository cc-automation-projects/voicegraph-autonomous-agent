.PHONY: install dev-up dev-down test lint typecheck build deploy integration-test dvc-pull dvc-repro clean

install:
	pip install -e ".[dev]"
	python -m spacy download en_core_web_lg
	python -m spacy download ru_core_news_sm
	pre-commit install

dev-up:
	docker-compose -f docker-compose.dev.yml up -d

dev-down:
	docker-compose -f docker-compose.dev.yml down -v

test:
	pytest tests/ -v --cov=src --cov-report=xml

integration-test:
	pytest tests/ -v -m "integration"

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

typecheck:
	mypy src/ --strict

build:
	docker-compose -f docker-compose.dev.yml build

deploy:
	kubectl apply -f infra/k8s/

dvc-pull:
	dvc pull

dvc-repro:
	dvc repro

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
