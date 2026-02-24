.PHONY: clean build check publish test install-dev

# Use .venv if present so tests run in isolated env (avoids system pytest plugin conflicts)
PYTHON := $(if $(wildcard .venv/bin/python),.venv/bin/python,python)
PIP := $(if $(wildcard .venv/bin/pip),.venv/bin/pip,pip)

# ==============================================================================
# Development
# ==============================================================================

install-dev:
	$(PYTHON) -m venv .venv 2>/dev/null || $(PYTHON) -m venv .venv
	$(PIP) install -e ".[dev]"

test:
	$(PYTHON) -m pytest tests -v -m "not slow"

test-unit:
	$(PYTHON) -m pytest tests/unit -v

# ==============================================================================
# Build and Publishing
# ==============================================================================

clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf dist/ build/ .eggs/ *.egg-info

build:
	@echo "🔨 Building package..."
	python -m build

check:
	@echo "✅ Checking distribution files..."
	twine check dist/*

publish: clean build check
	@echo "🚀 Publishing to PyPI..."
	twine upload dist/* 