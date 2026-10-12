# TODO

Planned work, roughly in priority order. Open an issue or pull request if you
want to pick one up.

## Before releasing 1.1.0

- [ ] Run one real document through each provider (Gemini, OpenAI, Anthropic);
      the test suite mocks all API calls
- [ ] Check that the Gemini default (`gemini-2.5-flash`) works with a newly
      created API key
- [ ] Publish to TestPyPI and install from there

## Release process

- [ ] Publish from GitHub Actions with PyPI Trusted Publishing instead of a
      local token
- [ ] Tag releases (`v1.1.0`) and create GitHub Releases from the changelog
- [ ] Keep the version in one place (it is duplicated in `pyproject.toml` and
      `llm_ocr/__init__.py`)

## Library

- [ ] Convert every page of multi-page TIFFs and every frame of GIFs (only the
      first is converted today)
- [ ] Detect truncated output from OpenAI (`finish_reason == "length"`) and
      Gemini, as the Anthropic provider already does
- [ ] Refresh the OpenAI and Gemini model constants and deprecate the ones for
      retired models
- [ ] Back off for longer on rate-limit (429) errors
- [ ] Ship a `py.typed` marker so users' type checkers see the type hints
- [ ] Optional automatic fallback model when Anthropic declines a request
