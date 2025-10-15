.PHONY: help install train eval serve test lint format clean

help:
	@echo "Available targets:"
	@echo "  install    - Install dependencies from requirements.txt"
	@echo "  train      - Train the churn prediction model"
	@echo "  eval       - Evaluate the trained model"
	@echo "  serve      - Start the FastAPI server"
	@echo "  test       - Run pytest"
	@echo "  lint       - Run code linting"
	@echo "  format     - Format code with black and isort"
	@echo "  clean      - Remove generated files"

install:
	pip install -r requirements.txt

train:
	python -m src.ml.train

eval:
	python -m src.ml.evaluate

serve:
	uvicorn src.api.app:app --reload --port 8000

test:
	pytest -q

lint:
	ruff check . || flake8 . || echo "No linter found"

format:
	black . && isort . || echo "Formatters not installed"

clean:
	rm -rf __pycache__ .pytest_cache models/*.joblib
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

