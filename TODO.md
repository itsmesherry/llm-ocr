# TODO

Planned work, roughly in priority order. Open an issue or pull request if you
want to pick one up.

## Library

- [ ] Convert every page of multi-page TIFFs and every frame of GIFs (only the
      first is converted today)
- [ ] Decide how to report blocked output from OpenAI (`content_filter`) and
      Gemini (`SAFETY`, `RECITATION`); those pages currently come back empty
- [ ] Refresh the OpenAI and Gemini model constants and deprecate the ones for
      retired models
- [ ] Do not retry errors that cannot succeed (invalid API key, 400/401/403);
      back off for longer on rate-limit (429) errors
- [ ] Ship a `py.typed` marker so users' type checkers see the type hints
- [ ] Optional automatic fallback model when Anthropic declines a request
