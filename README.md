# LLM OCR

[![PyPI](https://img.shields.io/pypi/v/ocr-llm)](https://pypi.org/project/ocr-llm/)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue)](https://pypi.org/project/ocr-llm/)
[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/Shehryar718/llm-ocr/blob/main/LICENSE)

Convert PDFs to markdown using Large Language Models (LLMs) with vision capabilities.

## Features

- 🔍 High-quality OCR using vision-capable LLMs
- 📄 Batch processing of multiple PDF pages
- 🔌 Multiple provider support (Gemini, OpenAI)
- ⚙️ Configurable processing settings
- 🔄 Automatic retry logic for transient errors
- 📝 Clean markdown output

## Installation

```bash
pip install ocr-llm
```

### System Dependencies

You also need to install poppler (required for PDF processing):

```bash
# macOS
brew install poppler

# Ubuntu/Debian
sudo apt-get install poppler-utils

# Fedora/RHEL
sudo yum install poppler-utils
```

### Dependencies

The library requires:

- **System**: `poppler-utils` for PDF processing
- **Python**:
  - `google-genai` for Gemini provider
  - `openai` for OpenAI provider
  - `pdf2image` and `Pillow` for PDF processing

## Quick Start

### Using OpenAI

```python
import asyncio
from llm_ocr import LLMOCR, OpenAI

async def main():
    # Initialize OpenAI provider
    provider = OpenAI(
        api_key="your-api-key",  # Or set OPENAI_API_KEY env var
        model=OpenAI.GPT_4O_MINI
    )

    # Create OCR processor
    async with LLMOCR(provider) as ocr:
        # Convert PDF to markdown
        markdown = await ocr.convert(
            "document.pdf",
            output_path="output.md"
        )
        print(markdown)

asyncio.run(main())
```

### Using Gemini

```python
import asyncio
from llm_ocr import LLMOCR, Gemini

async def main():
    # Initialize Gemini provider
    provider = Gemini(
        api_key="your-api-key",  # Or set GEMINI_API_KEY env var
        model=Gemini.FLASH_2_5  # Or Gemini.PRO_2_5 for best quality
    )

    # Create OCR processor
    async with LLMOCR(provider) as ocr:
        # Convert PDF to markdown
        markdown = await ocr.convert(
            "document.pdf",
            output_path="output.md"
        )
        print(markdown)

asyncio.run(main())
```

## Available Models

### OpenAI

- `OpenAI.GPT_4O`
- `OpenAI.GPT_4O_MINI` (default)

Additional models: `O1`, `O3`, `O4_MINI`, `GPT_5`, `GPT_5_MINI`, `GPT_4_1`, and more.

> See `llm_ocr/providers/openai.py` for the complete list.

### Gemini

- `Gemini.PRO_2_5`
- `Gemini.FLASH_2_5` (default)

Additional models: `PRO_2_0`, `FLASH_2_0`.

> See `llm_ocr/providers/gemini.py` for the complete list.

## Configuration

Customize the OCR processing with `OCRConfig`:

```python
from llm_ocr import LLMOCR, OpenAI, OCRConfig

config = OCRConfig(
    dpi=300,                    # Higher DPI for better quality
    max_pages=10,               # Limit number of pages to process
    llm_batch_size=2,           # Send 2 pages to LLM at once
    convert_to_grayscale=True,  # Convert images to grayscale
    max_retries=3,              # Retry failed requests
    retry_delay=1.0,            # Wait 1 second between retries
    include_page_markers=True,  # Add page markers in output
)

provider = OpenAI()
ocr = LLMOCR(provider, config=config)
```

### Configuration Options

| Option                 | Default | Description                                |
| ---------------------- | ------- | ------------------------------------------ |
| `dpi`                  | 200     | DPI for PDF to image conversion (72-600)   |
| `max_pages`            | None    | Maximum number of pages to process         |
| `batch_size`           | 5       | PDF to image conversion batch size         |
| `llm_batch_size`       | 1       | Number of pages to send to LLM at once     |
| `thread_count`         | 4       | Number of threads for PDF conversion       |
| `convert_to_grayscale` | False   | Convert images to grayscale                |
| `optimize_png`         | True    | Optimize PNG compression                   |
| `use_cropbox`          | True    | Use PDF cropbox for conversion             |
| `max_retries`          | 3       | Maximum retry attempts for failed requests |
| `retry_delay`          | 1.0     | Delay between retries in seconds           |
| `include_page_markers` | False   | Add page markers in markdown output        |

## Advanced Usage

### Custom Provider Parameters

Pass additional parameters to the LLM provider:

```python
# OpenAI with custom parameters
provider = OpenAI(
    model=OpenAI.GPT_4O,
    max_tokens=4000,
    temperature=0.0,
)

# Gemini with custom parameters
provider = Gemini(
    model=Gemini.PRO_2_5,
    temperature=0.0,
)
```

### Processing Multiple Documents

```python
import asyncio
from pathlib import Path
from llm_ocr import LLMOCR, OpenAI

async def process_documents():
    provider = OpenAI()

    async with LLMOCR(provider) as ocr:
        pdf_files = Path("pdfs").glob("*.pdf")

        for pdf_file in pdf_files:
            output_file = pdf_file.with_suffix(".md")
            await ocr.convert(pdf_file, output_path=output_file)
            print(f"Converted {pdf_file.name} -> {output_file.name}")

asyncio.run(process_documents())
```

### Without Context Manager

If you prefer not to use the context manager:

```python
import asyncio
from llm_ocr import LLMOCR, OpenAI

async def main():
    provider = OpenAI()
    ocr = LLMOCR(provider)

    try:
        markdown = await ocr.convert("document.pdf")
        print(markdown)
    finally:
        await ocr.aclose()  # Don't forget to close!

asyncio.run(main())
```

## Environment Variables

Set API keys via environment variables:

```bash
# For OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# For Gemini
export GEMINI_API_KEY="your-gemini-api-key"
```

Then use providers without passing API keys:

```python
# API key read from environment variable
provider = OpenAI()  # Uses OPENAI_API_KEY
# or
provider = Gemini()  # Uses GEMINI_API_KEY
```

## Error Handling

The library uses a fail-fast approach with automatic retries:

```python
import asyncio
from llm_ocr import LLMOCR, OpenAI, OCRConfig

async def main():
    provider = OpenAI()
    config = OCRConfig(
        max_retries=5,      # Retry up to 5 times
        retry_delay=2.0,    # Wait 2 seconds between retries
    )

    async with LLMOCR(provider, config) as ocr:
        try:
            markdown = await ocr.convert("document.pdf")
            print(markdown)
        except Exception as e:
            print(f"Failed to process document: {e}")

asyncio.run(main())
```

## License

See LICENSE file for details.
