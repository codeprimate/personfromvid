# How to Publish to PyPI

This guide outlines the steps to publish the `personfromvid` package to the Python Package Index (PyPI).

## Prerequisites

Before you begin, ensure you have the necessary tools installed and accounts created.

### 1. Install Publishing Tools

You need `build` to create the package distribution files and `twine` to securely upload them.

```bash
pip install build twine
```

### 2. Create PyPI Accounts

You will need accounts on both the TestPyPI server (for practice runs) and the official PyPI server.

- **TestPyPI**: [https://test.pypi.org/account/register/](https://test.pypi.org/account/register/)
- **PyPI**: [https://pypi.org/account/register/](https://pypi.org/account/register/)

Using TestPyPI first is strongly recommended to verify that your package works correctly before a public release.

### 3. Create API Tokens

For security, use API tokens to upload your packages.

1.  Log in to your TestPyPI and PyPI accounts.
2.  Navigate to "Account settings" -> "API tokens".
3.  Create a new token. You can scope it to the `personfromvid` project if you wish.
4.  **Important**: Copy the token immediately. You will not be able to view it again.

When `twine` prompts for a username, enter `__token__`. For the password, paste the entire API token (including the `pypi-` prefix).

## Publishing Steps

### Step 1: Update Version Number

Before publishing a new release, you must increment the version number in `pyproject.toml`. PyPI does not allow overwriting existing versions.

```toml
# pyproject.toml
[project]
name = "personfromvid"
version = "1.0.1" # Changed from 1.0.0
# ...
```

### Step 2: Build the Package

From the root of the project, run the build command. This will generate the distribution archives (`.whl` and `.tar.gz`) in a `dist/` directory.

```bash
# Optional: Remove old build artifacts
rm -rf dist/

# Build the package
python -m build
```

### Step 3: Check the Distribution Files

Use `twine` to validate that your package metadata will render correctly on PyPI.

```bash
twine check dist/*
```

### Step 4: Upload to TestPyPI

Upload your package to the test server to ensure the process works as expected.

```bash
twine upload --repository testpypi dist/*
```

Provide `__token__` as the username and your **TestPyPI API token** as the password.

### Step 5: Test the Installation from TestPyPI

Install the package from TestPyPI to confirm it works correctly. The `--extra-index-url` flag allows `pip` to find dependencies on the official PyPI, since they won't be on the test server.

```bash
pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple \
  personfromvid
```

After installation, verify that the command-line script is working:

```bash
personfromvid --version
```

### Step 6: Upload to PyPI

Once you have confirmed that the package installs and runs correctly from TestPyPI, you are ready to publish to the official PyPI repository.

```bash
twine upload dist/*
```

Use `__token__` as the username and your **official PyPI API token** as the password.

Congratulations, your package is now live on PyPI!

## Future Updates

For all subsequent releases, follow this checklist:

1.  Ensure all code changes are merged and tests are passing.
2.  Increment the `version` in `pyproject.toml`.
3.  Re-run the build process (`python -m build`).
4.  Upload the new version using `twine upload dist/*`.