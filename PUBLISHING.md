# Publishing to PyPI

This guide explains how to publish the `llm-ocr` package to PyPI.

## Prerequisites

1. Install build tools:

```bash
pip install build twine
```

2. Create a PyPI account at https://pypi.org/account/register/

3. Generate an API token at https://pypi.org/manage/account/token/

## Building the Package

1. Clean previous builds (if any):

```bash
rm -rf dist/ build/ *.egg-info
```

2. Build the package:

```bash
python -m build
```

This will create two files in the `dist/` directory:

- `llm_ocr-1.0.0-py3-none-any.whl` (wheel distribution)
- `llm-ocr-1.0.0.tar.gz` (source distribution)

## Publishing to PyPI

### Option 1: Using Twine (Recommended)

Upload to PyPI:

```bash
python -m twine upload dist/*
```

You'll be prompted for:

- Username: `__token__`
- Password: Your PyPI API token (including the `pypi-` prefix)

### Option 2: Using .pypirc

1. Create a `~/.pypirc` file (see `.pypirc.example` in this repo):

```ini
[distutils]
index-servers =
    pypi

[pypi]
username = __token__
password = pypi-YOUR_API_TOKEN_HERE
```

2. Upload:

```bash
python -m twine upload dist/*
```

### Testing on TestPyPI First (Optional but Recommended)

Before publishing to the real PyPI, test on TestPyPI:

1. Create an account at https://test.pypi.org/

2. Upload to TestPyPI:

```bash
python -m twine upload --repository testpypi dist/*
```

3. Test installation:

```bash
pip install --index-url https://test.pypi.org/simple/ llm-ocr
```

## Verification

After publishing, verify the package:

1. Check the package page: https://pypi.org/project/llm-ocr/

2. Install and test:

```bash
pip install llm-ocr
python -c "import llm_ocr; print(llm_ocr.__version__)"
```

## Updating the Version

For future releases:

1. Update version in `pyproject.toml`
2. Update version in `llm_ocr/__init__.py`
3. Clean, build, and upload again

## Troubleshooting

### "File already exists" error

This means the version is already published. You need to increment the version number.

### Import errors

Make sure all dependencies are properly listed in `pyproject.toml` under `dependencies`.

### Missing files in distribution

Check `MANIFEST.in` to ensure all necessary files are included.

## Quick Command Reference

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build
python -m build

# Check distribution
python -m twine check dist/*

# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# Upload to PyPI
python -m twine upload dist/*
```
