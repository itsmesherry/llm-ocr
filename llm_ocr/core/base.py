from abc import ABC, abstractmethod
from typing import List


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, api_key: str, model: str, **kwargs):
        """
        Initialize the LLM provider.

        Args:
            api_key: API key for the provider
            model: Model identifier/name
            **kwargs: Additional provider-specific parameters
        """
        self.api_key = api_key
        self.model = model
        self.kwargs = kwargs

    @abstractmethod
    async def generate_content(
        self,
        prompt: str,
        image_bytes_list: List[bytes],
    ) -> str:
        """
        Generate content from images and prompt.

        Args:
            prompt: The instruction prompt
            image_bytes_list: List of PNG image bytes (one or more pages)

        Returns:
            Generated text content
        """
        pass

    @abstractmethod
    async def aclose(self):
        """Close and cleanup resources."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name."""
        pass
