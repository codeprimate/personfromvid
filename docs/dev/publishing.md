# How to Publish to PyPI

This guide outlines the simplified steps to publish the `personfromvid` package to the Python Package Index (PyPI) using a `Makefile`.

## Prerequisites

### 1. Install Publishing Tools

You need `build` to create the package and `twine` to upload it. If you don't have them, install them:

```bash
pip install build twine
```

### 2. Get a PyPI API Token

You need an account on the official PyPI server to publish.

- **PyPI Website**: [https://pypi.org/account/register/](https://pypi.org/account/register/)

After creating an account, generate an API token for authentication.

1.  Log in to your PyPI account.
2.  Navigate to "Account settings" -> "API tokens".
3.  Create a new token. It's good practice to scope it to the `personfromvid` project.
4.  **Important**: Copy the token immediately, as you will not be able to view it again.

## Publishing Steps

The publishing process is automated with a `Makefile`.

### Step 1: Update Version Number

Before publishing a new release, you must increment the `version` number in `pyproject.toml`. PyPI does not allow overwriting existing versions.

```toml
# pyproject.toml
[project]
name = "personfromvid"
version = "1.0.1" # Example: changed from 1.0.0
# ...
```

### Step 2: Publish the Package

Run the `publish` command from the root of the project.

```bash
make publish
```

This single command will:
1.  Clean any old build artifacts.
2.  Build the source and wheel distributions.
3.  Check the new distribution files.
4.  Upload the package to PyPI.

### Authentication

When you run `make publish`, `twine` will prompt you for your credentials.
-   **Username**: Enter `__token__`
-   **Password**: Paste your PyPI API token (including the `pypi-` prefix).

#### Optional: Using Environment Variables

For a non-interactive workflow, you can configure `twine` by setting environment variables in your shell.

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD="pypi-your-token-here"
```

If these variables are set, `twine` will use them to authenticate automatically and will not prompt for your credentials.

## Future Updates

For all subsequent releases, the process is simple:

1.  Ensure all code changes are merged and tests are passing.
2.  Increment the `version` in `pyproject.toml`.
3.  Run `make publish`.