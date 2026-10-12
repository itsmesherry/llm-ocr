# Contributing

Thanks for helping improve `ocr-llm`. Bug reports, fixes and new providers are all welcome.

## Setup

```bash
git clone https://github.com/itsmesherry/llm-ocr.git
cd llm-ocr
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

PDF conversion needs `poppler` (`brew install poppler` or `apt install poppler-utils`).
The unit tests mock it, so it is only required to run the library on real PDFs.

## Before opening a pull request

```bash
make qa          # isort, ruff format, ruff check
mypy llm_ocr
make test
```

CI runs the same checks on Python 3.9 to 3.13, so avoid syntax newer than 3.9
(for example, write `Optional[str]`, not `str | None`).

A pull request should:

- include a test for any behaviour it adds or fixes; providers are mocked, so no
  API key is needed
- add a line under `## [Unreleased]` in [CHANGELOG.md](CHANGELOG.md)
- keep existing public names working; see below

## Compatibility

People depend on this package, so within a major version:

- do not remove or rename public classes, methods, `OCRConfig` fields or model
  constants
- new `OCRConfig` fields need a default that preserves current behaviour
- when a provider retires a model, mark its constant as deprecated in the
  changelog for one minor release before removing it

## Adding a provider

Subclass `LLMProvider` in `llm_ocr/providers/`, implement `generate_content`,
`aclose` and `name`, and export it from `llm_ocr/__init__.py`. If it needs a new
SDK, add it as an optional extra in `pyproject.toml`, not as a core dependency.
Raise `OCRResponseError` when the model returns truncated or refused output.

## Releasing (maintainers)

1. Open a release pull request that bumps the version in `pyproject.toml` and
   `llm_ocr/__init__.py`, and renames `Unreleased` in `CHANGELOG.md` to the new
   version and date (leave a fresh, empty `Unreleased` above it).
2. Merge it once CI is green.
3. Create a GitHub Release on that commit with the tag `v<version>`, for example
   `v1.2.0`, and paste the changelog section as the notes.

Publishing the release runs `.github/workflows/publish.yml`, which builds the
package from the tag and uploads it to PyPI through Trusted Publishing. No API
token is involved, and the workflow refuses to publish if the tag does not match
the version in `pyproject.toml`.

To rehearse an upload, `make publish-test` sends a local build to TestPyPI.
