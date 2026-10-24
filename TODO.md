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
- [ ] Back off for longer on rate-limit (429) errors
- [ ] Ship a `py.typed` marker so users' type checkers see the type hints
- [ ] Optional automatic fallback model when Anthropic declines a request
- [ ] Detect pages where the model loops on the same line and retry them
      instead of returning the repeated text
- [ ] Default `temperature` to 0 for every provider, overridable by the user

## Documentation

- [ ] Recommended local models table (e.g. `qwen2.5vl`, OCR-tuned VLMs) with
      the serving settings used
- [ ] Benchmark one PDF on Ollama at `max_concurrency` 1, 2 and 4: total time,
      peak VRAM, retried pages; note how `OLLAMA_NUM_PARALLEL` affects it
