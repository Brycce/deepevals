import time
from typing import List, Dict
import httpx

from app.providers.base import ModelProvider, GenerationResult
from app.config import settings


class GroqProvider(ModelProvider):
    """Provider for Groq models (fast inference)."""

    BASE_URL = "https://api.groq.com/openai/v1/chat/completions"

    MODELS = [
        {"id": "openai/gpt-oss-120b", "display_name": "GPT-OSS 120B"},
        {"id": "openai/gpt-oss-20b", "display_name": "GPT-OSS 20B"},
        {"id": "qwen/qwen3-32b", "display_name": "Qwen 3 32B"},
        {"id": "llama-3.3-70b-versatile", "display_name": "Llama 3.3 70B"},
        {"id": "llama-3.1-8b-instant", "display_name": "Llama 3.1 8B Instant"},
        {"id": "meta-llama/llama-4-maverick-17b-128e-instruct", "display_name": "Llama 4 Maverick 17B"},
        {"id": "meta-llama/llama-4-scout-17b-16e-instruct", "display_name": "Llama 4 Scout 17B"},
        {"id": "moonshotai/kimi-k2-instruct", "display_name": "Kimi K2"},
    ]

    def __init__(self):
        self.api_key = settings.GROQ_API_KEY

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def get_available_models(self) -> List[Dict[str, str]]:
        return [{"provider": "groq", **m} for m in self.MODELS]

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
                error="Groq API key not configured"
            )

        start_time = time.time()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
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
