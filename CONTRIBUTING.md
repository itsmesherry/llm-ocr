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

1. Move the `Unreleased` entries in `CHANGELOG.md` under a new version heading.
2. Bump the version in `pyproject.toml` and `llm_ocr/__init__.py`.
3. `make publish-test`, install from TestPyPI and try it.
4. `make publish`, then tag the commit (`git tag v1.2.0 && git push --tags`).
