from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict


@dataclass
class GenerationResult:
    """Result from a model generation."""
    output_text: str
    input_tokens: int
    output_tokens: int
    generation_time_ms: int
    error: Optional[str] = None


class ModelProvider(ABC):
    """Abstract base class for model providers."""

    @abstractmethod
    async def generate(
        self,
        model_id: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4000,
    ) -> GenerationResult:
        """Generate a response from the model."""
        pass

    @abstractmethod
    def get_available_models(self) -> List[Dict[str, str]]:
        """Return list of available models for this provider."""
        pass

    @abstractmethod
    def is_configured(self) -> bool:
        """Check if the provider has valid API credentials."""
        pass
