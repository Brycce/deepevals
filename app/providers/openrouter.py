import time
from typing import List, Dict
import httpx

from app.providers.base import ModelProvider, GenerationResult
from app.config import settings


class OpenRouterProvider(ModelProvider):
    """Provider for OpenRouter models (DeepSeek, Llama, Grok, Kimi)."""

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    MODELS = [
        {"id": "deepseek/deepseek-chat", "display_name": "DeepSeek Chat"},
        {"id": "meta-llama/llama-3.3-70b-instruct", "display_name": "Llama 3.3 70B"},
        {"id": "x-ai/grok-2-1212", "display_name": "Grok 2"},
        {"id": "moonshotai/moonshot-v1-128k", "display_name": "Kimi"},
    ]

    def __init__(self):
        self.api_key = settings.OPENROUTER_API_KEY

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def get_available_models(self) -> List[Dict[str, str]]:
        return [{"provider": "openrouter", **m} for m in self.MODELS]

    async def generate(
        self,
        model_id: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4000,
    ) -> GenerationResult:
        if not self.api_key:
            return GenerationResult(
                output_text="",
                input_tokens=0,
                output_tokens=0,
                generation_time_ms=0,
                error="OpenRouter API key not configured"
            )

        start_time = time.time()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://deepevals.local",
            "X-Title": "DeepEvals"
        }

        payload = {
            "model": model_id,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    self.BASE_URL,
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                data = response.json()

            generation_time_ms = int((time.time() - start_time) * 1000)

            output_text = data["choices"][0]["message"]["content"] or ""
            usage = data.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)

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
