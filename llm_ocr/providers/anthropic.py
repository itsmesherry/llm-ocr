import base64
import logging
import os
from typing import Any, List, Optional

try:
    from anthropic import AsyncAnthropic
except ImportError:  # pragma: no cover
    AsyncAnthropic = None  # type: ignore[assignment,misc]

from ..core.base import LLMProvider, OCRResponseError

logger = logging.getLogger(__name__)


class Anthropic(LLMProvider):
    """
    Anthropic Claude provider for OCR.

    Requires the optional dependency: pip install "ocr-llm[anthropic]"

    Supported models:
    - claude-sonnet-5-5 (default, balanced)
    - claude-opus-5-5 (best quality)
    - claude-haiku-5-5 (fastest, cheapest)

    Any other model identifier can be passed as a plain string.
    """

    # Available models with their identifiers
    CLAUDE_OPUS_5_5 = "claude-opus-5-5"
    CLAUDE_SONNET_5_5 = "claude-sonnet-5-5"
    CLAUDE_HAIKU_5_5 = "claude-haiku-5-5"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = CLAUDE_SONNET_5_5,
        max_tokens: int = 64000,
        **kwargs,
    ):
        """
        Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var
            model: Model to use. Use Anthropic.CLAUDE_SONNET_5_5, etc.
            max_tokens: Maximum tokens to generate per request (default: 64000).
                This is a ceiling, not a target; it also covers the model's
                thinking, so keep it generous.
            **kwargs: Additional parameters for the Anthropic Messages API
                (e.g., output_config={"effort": "low"})
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
            # Streamed so that large outputs don't hit HTTP timeouts
            async with self.aclient.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                **self.kwargs,
            ) as stream:
                response = await stream.get_final_message()
        except Exception as err:
            logger.error("Anthropic generation failed: %s", err)
            raise

        if response.stop_reason == "max_tokens":
            raise OCRResponseError(
                f"Anthropic output was truncated at max_tokens={self.max_tokens}. "
                "Increase max_tokens or lower llm_batch_size."
            )
        if response.stop_reason == "refusal":
            raise OCRResponseError("Anthropic declined to process this content.")

        text_parts = [block.text for block in response.content if block.type == "text"]
        if not text_parts:
            logger.warning("Anthropic returned no content")
        return "\n".join(text_parts)

    async def aclose(self):
        """Close the Anthropic client."""
        await self.aclient.close()
