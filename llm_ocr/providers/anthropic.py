import base64
import logging
import os
from typing import Any, List, Optional

try:
    from anthropic import AsyncAnthropic
except ImportError:  # pragma: no cover
    AsyncAnthropic = None  # type: ignore[assignment,misc]

from ..core.base import LLMProvider

logger = logging.getLogger(__name__)


class Anthropic(LLMProvider):
    """
    Anthropic Claude provider for OCR.

    Supported models:
    - claude-sonnet-4-20250514 (default, recommended)
    - claude-opus-4-20250514
    - claude-3-7-sonnet-latest
    - claude-3-5-sonnet-latest
    - claude-3-5-haiku-latest
    """

    # Available models with their identifiers
    CLAUDE_SONNET_4 = "claude-sonnet-4-20250514"
    CLAUDE_OPUS_4 = "claude-opus-4-20250514"
    CLAUDE_3_7_SONNET = "claude-3-7-sonnet-latest"
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet-latest"
    CLAUDE_3_5_HAIKU = "claude-3-5-haiku-latest"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = CLAUDE_SONNET_4,
        max_tokens: int = 4096,
        **kwargs,
    ):
        """
        Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var
            model: Model to use. Use Anthropic.CLAUDE_SONNET_4, etc.
            max_tokens: Maximum tokens to generate per request (default: 4096)
            **kwargs: Additional parameters for the Anthropic API (e.g., temperature)
        """
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key is required. "
                "Pass it directly or set ANTHROPIC_API_KEY environment variable."
            )

        if AsyncAnthropic is None:
            raise ImportError(
                "The 'anthropic' package is required to use the Anthropic provider. "
                "Install it with: pip install 'ocr-llm[anthropic]'"
            )

        self.max_tokens = max_tokens
        super().__init__(api_key, model, **kwargs)

        # Initialize the async client
        self.aclient = AsyncAnthropic(api_key=self.api_key)

    @property
    def name(self) -> str:
        """Return provider name."""
        return "anthropic"

    async def generate_content(
        self,
        prompt: str,
        image_bytes_list: List[bytes],
    ) -> str:
        """Generate content using Anthropic Claude with one or more images."""
        content: List[Any] = []

        # Add all images as base64-encoded blocks
        for image_bytes in image_bytes_list:
            b64_image = base64.b64encode(image_bytes).decode("utf-8")
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": b64_image,
                    },
                }
            )

        # Add the text prompt
        content.append({"type": "text", "text": prompt})

        try:
            response = await self.aclient.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                **self.kwargs,
            )

            if response.content and len(response.content) > 0:
                text_parts = [
                    block.text
                    for block in response.content
                    if hasattr(block, "text") and block.text
                ]
                if text_parts:
                    return "\n".join(text_parts)
            logger.warning("Anthropic returned no content")
            return ""

        except Exception as err:
            logger.error("Anthropic generation failed: %s", err)
            raise

    async def aclose(self):
        """Close the Anthropic client."""
        await self.aclient.close()
