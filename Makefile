.PHONY: install install-dev install-api install-app train-text train-image evaluate-text evaluate-image lint test clean

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev]"

install-api:
	pip install -r requirements.txt
	pip install -e ".[api]"

install-app:
	pip install -r requirements.txt
	pip install -e ".[app]"

train-text:
	python -m src.text_classifier.train --data data/text/sample_data.csv

train-image:
	python -m src.image_classifier.train --data data/image

evaluate-text:
	python -m src.text_classifier.evaluate

evaluate-image:
	python -m src.image_classifier.evaluate

lint:
	ruff check .

format:
	ruff format .

typecheck:
	mypy src/

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=term-missing

run-text-app:
	python -m app.text_app

run-image-app:
	python -m app.image_app

run-api:
	cd api && uvicorn main:app --reload

clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf src/text_classifier/model
	rm -rf src/image_classifier/model
	rm -rf outputs/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

help:
	@echo "Available commands:"
	@echo "  make install          - Install dependencies"
	@echo "  make install-dev     - Install with dev dependencies"
	@echo "  make install-api     - Install with API dependencies"
	@echo "  make install-app     - Install with Gradio app dependencies"
	@echo "  make train-text      - Train text classifier"
	@echo "  make train-image     - Train image classifier"
	@echo "  make evaluate-text   - Evaluate text classifier"
	@echo "  make evaluate-image  - Evaluate image classifier"
	@echo "  make lint            - Run linter"
	@echo "  make format          - Format code"
	@echo "  make typecheck       - Run type checker"
	@echo "  make test            - Run tests"
	@echo "  make test-cov        - Run tests with coverage"
	@echo "  make run-text-app    - Run text classifier Gradio app"
	@echo "  make run-image-app   - Run image classifier Gradio app"
	@echo "  make run-api         - Run FastAPI server"
	@echo "  make clean           - Clean up generated files"
