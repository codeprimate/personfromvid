.PHONY: clean build check publish

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