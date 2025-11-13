# Makefile for AI Development Roadmap

.PHONY: help install sync run test format lint clean

help:
	@echo "AI Development Roadmap - Available Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install    - Install uv package manager"
	@echo "  make sync       - Install all dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make run-numpy  - Run NumPy examples"
	@echo "  make run-phase1 - Run all Phase 1 examples"
	@echo "  make test       - Run all tests"
	@echo ""
	@echo "Code Quality:"
	@echo "  make format     - Format code with ruff"
	@echo "  make lint       - Lint code with ruff"
	@echo "  make type-check - Type check with mypy"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean      - Remove cache and temp files"
	@echo "  make status     - Show learning progress"

install:
	@echo "Installing uv package manager..."
	curl -LsSf https://astral.sh/uv/install.sh | sh

sync:
	@echo "Syncing dependencies..."
	uv sync

run-numpy:
	@echo "Running NumPy examples..."
	uv run python phase1_foundations/01_numpy_basics/examples.py

run-phase1:
	@echo "Running Phase 1 examples..."
	uv run python phase1_foundations/01_numpy_basics/examples.py

test:
	@echo "Running tests..."
	uv run pytest tests/ -v

format:
	@echo "Formatting code..."
	uv run ruff format .

lint:
	@echo "Linting code..."
	uv run ruff check .

type-check:
	@echo "Type checking..."
	uv run mypy .

clean:
	@echo "Cleaning cache and temp files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf htmlcov/
	rm -rf .coverage

status:
	@echo "Learning Progress:"
	@echo ""
	@cat .serena/memories/learning_progress.md | grep -E "^- \[" | head -20
