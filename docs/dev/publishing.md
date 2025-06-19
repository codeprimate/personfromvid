# Publishing to PyPI

This guide covers how to publish the `personfromvid` package to the Python Package Index (PyPI). The process is automated using a `Makefile` for simplicity and consistency.

## Overview

Publishing a new release involves three main steps:
1. **Update the version** in `pyproject.toml`
2. **Create a git tag** to mark the release
3. **Run the publish command** to upload to PyPI

After initial setup, each release follows the same simple workflow.

## Release Checklist

- [ ] All code changes merged and tests passing
- [ ] Update version number in `pyproject.toml`
- [ ] Commit version change: `git commit -m "Bump version to X.X.X"`
- [ ] Create and push git tag: `git tag vX.X.X && git push origin main && git push origin vX.X.X`
- [ ] Run publish command: `make publish`
- [ ] Verify package published at [https://pypi.org/project/personfromvid/](https://pypi.org/project/personfromvid/)

## One-Time Setup

These steps only need to be done once when setting up publishing for the first time.

### 1. Install Dependencies

Ensure all project dependencies are installed, including the publishing tools (`build` and `twine`):

```bash
pip install -r requirements.txt
```

### 2. Create PyPI Account and API Token

You need an account on PyPI to publish packages.

1. **Create an account**: [https://pypi.org/account/register/](https://pypi.org/account/register/)
2. **Generate an API token**:
   - Log in to your PyPI account
   - Navigate to "Account settings" → "API tokens"
   - Create a new token (scope it to the `personfromvid` project if desired)
   - **Important**: Copy the token immediately, as you cannot view it again

## Release Process

Follow these steps for each new release:

### Step 1: Update Version Number

Increment the `version` number in `pyproject.toml`. PyPI does not allow overwriting existing versions.

```toml
# pyproject.toml
[project]
name = "personfromvid"
version = "1.0.1" # Example: changed from 1.0.0
# ...
```

### Step 2: Create Git Tag

Commit the version change and create a git tag to mark the release:

```bash
git add pyproject.toml
git commit -m "Bump version to 1.0.1"
git tag v1.0.1
git push origin main
git push origin v1.0.1
```

**Note**: Replace `1.0.1` with your actual version number. Tags should follow the `v{version}` format.

### Step 3: Publish to PyPI

Run the automated publish command:

```bash
make publish
```

This command will:
1. Clean any old build artifacts
2. Build the source and wheel distributions
3. Check the new distribution files
4. Upload the package to PyPI

## Authentication Reference

### Interactive Authentication

When you run `make publish`, `twine` will prompt for credentials:
- **Username**: Enter `__token__`
- **Password**: Paste your PyPI API token (including the `pypi-` prefix)

### Non-Interactive Authentication (Optional)

For automated workflows, you can set environment variables:

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD="pypi-your-token-here"
```

With these variables set, `twine` will authenticate automatically without prompting.

## Quick Reference

For subsequent releases after initial setup:

1. Ensure all code changes are merged and tests are passing
2. Increment the `version` in `pyproject.toml`
3. Create a git tag for the new version
4. Run `make publish`