import time
from typing import List, Dict
import anthropic

from app.providers.base import ModelProvider, GenerationResult
from app.config import settings


class AnthropicProvider(ModelProvider):
    """Provider for Anthropic Claude models."""

    MODELS = [
        {"id": "claude-sonnet-4-5-20250929", "display_name": "Claude Sonnet 4.5"},
        {"id": "claude-haiku-4-5-20251001", "display_name": "Claude Haiku 4.5"},
    ]

    def __init__(self):
        self.api_key = settings.ANTHROPIC_API_KEY
        if self.api_key:
            self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
        else:
            self.client = None

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def get_available_models(self) -> List[Dict[str, str]]:
        return [{"provider": "anthropic", **m} for m in self.MODELS]

    async def generate(
        self,
        model_id: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4000,
    ) -> GenerationResult:
        if not self.client:
            return GenerationResult(
                output_text="",
                input_tokens=0,
                output_tokens=0,
                generation_time_ms=0,
                error="Anthropic API key not configured"
            )

        start_time = time.time()

        try:
            response = await self.client.messages.create(
                model=model_id,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )

            generation_time_ms = int((time.time() - start_time) * 1000)

            output_text = ""
            for block in response.content:
                if block.type == "text":
                    output_text += block.text

            return GenerationResult(
                output_text=output_text,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                generation_time_ms=generation_time_ms,
            )

        except Exception as e:
            generation_time_ms = int((time.time() - start_time) * 1000)
            return GenerationResult(
                output_text="",
                input_tokens=0,
                output_tokens=0,
                generation_time_ms=generation_time_ms,
                error=str(e)
            )
