from dataclasses import dataclass
from typing import Optional


@dataclass
class OCRConfig:
    """Configuration for OCR processing."""

    dpi: int = 200
    max_pages: Optional[int] = None
    batch_size: int = 5  # PDF to image conversion batch size
    llm_batch_size: int = 1  # Number of pages to send to LLM at once
    thread_count: int = 4
    convert_to_grayscale: bool = False
    optimize_png: bool = True
    use_cropbox: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0
    output_format: str = "markdown"
    include_page_markers: bool = False

    def validate(self):
        """Validate configuration."""
        if self.dpi < 72 or self.dpi > 600:
            raise ValueError("DPI must be between 72 and 600")
        if self.batch_size < 1:
            raise ValueError("Batch size must be at least 1")
        if self.llm_batch_size < 1:
            raise ValueError("LLM batch size must be at least 1")
