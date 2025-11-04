import base64
import os
from typing import List, Optional

from openai import AsyncOpenAI

from ..core.base import LLMProvider


class OpenAI(LLMProvider):
    """
    OpenAI provider for OCR.

    Supported models:
    - gpt-5
    - gpt-5-mini
    - gpt-5-nano
    - gpt-5-chat-latest
    - gpt-4.1
    - gpt-4.1-mini
    - gpt-4.1-nano
    - o4-mini
    - o1
    - o3
    - gpt-4o
    - gpt-4o-mini
    """

    # Available models with their identifiers
    GPT_5 = "gpt-5"
    GPT_5_MINI = "gpt-5-mini"
    GPT_5_NANO = "gpt-5-nano"
    GPT_5_CHAT_LATEST = "gpt-5-chat-latest"

    GPT_4_1 = "gpt-4.1"
    GPT_4_1_MINI = "gpt-4.1-mini"
    GPT_4_1_NANO = "gpt-4.1-nano"

    O4_MINI = "o4-mini"
    O1 = "o1"
    O3 = "o3"

    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"

    def __init__(
        self, api_key: Optional[str] = None, model: str = GPT_4O_MINI, **kwargs
    ):
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var
            model: Model to use. Use OpenAI.GPT_4O, OpenAI.GPT_4O_MINI, etc.
            **kwargs: Additional parameters for the OpenAI API (e.g., max_tokens, temperature)
        """
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. "
                "Pass it directly or set OPENAI_API_KEY environment variable."
            )

        super().__init__(api_key, model, **kwargs)

        # Initialize the async client
        self.aclient = AsyncOpenAI(api_key=self.api_key)

    @property
    def name(self) -> str:
        """Return provider name."""
        return "openai"

    async def generate_content(
        self,
        prompt: str,
        image_bytes_list: List[bytes],
    ) -> str:
        """Generate content using OpenAI with one or more images."""
        # Create content array with text prompt
        content = [{"type": "text", "text": prompt}]

        # Add all images as base64-encoded data URLs
        for image_bytes in image_bytes_list:
            b64_image = base64.b64encode(image_bytes).decode("utf-8")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64_image}",
                    },
                }
            )

        try:
            # Use the Chat Completions API with vision
            response = await self.aclient.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                **self.kwargs,
            )

            # Extract the text content from the response
            if response.choices and len(response.choices) > 0:
                message_content = response.choices[0].message.content
                return message_content or ""
            else:
                print("[WARNING] OpenAI returned no content")
                return ""

        except Exception as err:
            print(f"[ERROR] OpenAI generation failed: {err}")
            raise

    async def aclose(self):
        """Close the OpenAI client."""
        # Close the async client
        await self.aclient.close()
