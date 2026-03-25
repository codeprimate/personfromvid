.PHONY: clean build check publish test install-dev

# ==============================================================================
# Development
# ==============================================================================

install-dev:
	uv sync --extra dev

test:
	uv run pytest tests -v -m "not slow"

test-unit:
	uv run pytest tests/unit -v

# ==============================================================================
# Build and Publishing
# ==============================================================================

clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf dist/ build/ .eggs/ *.egg-info

build:
	@echo "🔨 Building package..."
	uv build

check:
	@echo "✅ Checking distribution files..."
	uvx twine check dist/*

publish: clean build check
	@echo "🚀 Publishing to PyPI..."
	uvx twine upload dist/*
