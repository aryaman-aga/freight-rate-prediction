# Makefile for Freight Rate Prediction

.PHONY: help setup train predict score test lint clean all

# Default target
all: setup train predict score

help:
	@echo "Freight Rate Prediction - Available targets:"
	@echo "  setup     - Create venv and install dependencies"
	@echo "  train     - Train models with CV"
	@echo "  predict   - Generate validation_predictions.csv and fill december_chart_inputs.csv"
	@echo "  score     - Run score.py to validate outputs and create chart"
	@echo "  test      - Run pytest suite"
	@echo "  lint      - Run ruff linter"
	@echo "  clean     - Remove generated artifacts"
	@echo "  all       - Run full pipeline (setup train predict score)"

# Python interpreter from venv
PYTHON := .venv/bin/python

setup:
	@echo "Creating virtual environment..."
	python3 -m venv .venv
	@echo "Installing dependencies..."
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	@echo "Setup complete. Activate with: source .venv/bin/activate"

train:
	@echo "Training models..."
	$(PYTHON) -m freight_rate.train

predict:
	@echo "Generating predictions..."
	$(PYTHON) -m freight_rate.predict

score:
	@echo "Running scorer..."
	$(PYTHON) score.py --predictions output/validation_predictions.csv --december-predictions data/december_chart_inputs.csv

test:
	@echo "Running tests..."
	$(PYTHON) -m pytest tests/ -v

lint:
	@echo "Running ruff..."
	$(PYTHON) -m ruff check src/ tests/
	$(PYTHON) -m ruff format --check src/ tests/

clean:
	@echo "Cleaning artifacts..."
	rm -rf output/ scorer_results/ models/ reports/*.png reports/*.json
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# Development helpers
notebook:
	$(PYTHON) -m jupyter lab notebooks/

eda:
	$(PYTHON) notebooks/01_eda.ipynb