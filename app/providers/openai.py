import time
from typing import List, Dict
from openai import AsyncOpenAI

from app.providers.base import ModelProvider, GenerationResult
from app.config import settings


class OpenAIProvider(ModelProvider):
    """Provider for OpenAI models."""

    MODELS = [
        {"id": "gpt-4", "display_name": "GPT-4"},
        {"id": "gpt-4o", "display_name": "GPT-4o"},
    ]

    def __init__(self):
        self.api_key = settings.OPENAI_API_KEY
        if self.api_key:
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                timeout=120.0  # 2 minute timeout
            )
        else:
            self.client = None

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def get_available_models(self) -> List[Dict[str, str]]:
        return [{"provider": "openai", **m} for m in self.MODELS]

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
                error="OpenAI API key not configured"
            )

        start_time = time.time()

        try:
            response = await self.client.chat.completions.create(
                model=model_id,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )

            generation_time_ms = int((time.time() - start_time) * 1000)

            output_text = response.choices[0].message.content or ""
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0

            return GenerationResult(
                output_text=output_text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
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
