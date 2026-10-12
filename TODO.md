# TODO

Planned work, roughly in priority order. Open an issue or pull request if you
want to pick one up.

## Release process

- [ ] Publish from GitHub Actions with PyPI Trusted Publishing instead of a
      local token
- [ ] Publish to TestPyPI before each release (skipped for 1.1.0)
- [ ] Keep the version in one place (it is duplicated in `pyproject.toml` and
      `llm_ocr/__init__.py`)

## Library

- [ ] Convert every page of multi-page TIFFs and every frame of GIFs (only the
      first is converted today)
- [ ] Detect truncated output from OpenAI (`finish_reason == "length"`) and
      Gemini, as the Anthropic provider already does
- [ ] Refresh the OpenAI and Gemini model constants and deprecate the ones for
      retired models
- [ ] Do not retry errors that cannot succeed (invalid API key, 400/401/403);
      back off for longer on rate-limit (429) errors
- [ ] Ship a `py.typed` marker so users' type checkers see the type hints
- [ ] Optional automatic fallback model when Anthropic declines a request
