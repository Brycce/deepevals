from pydantic_settings import BaseSettings
from typing import List
import os


class ModelConfig:
    """Configuration for a single model."""
    def __init__(self, provider: str, id: str, display_name: str):
        self.provider = provider
        self.id = id
        self.display_name = display_name

    def to_dict(self):
        return {
            "provider": self.provider,
            "id": self.id,
            "display_name": self.display_name
        }


class Settings(BaseSettings):
    # Database - supports SQLite locally or PostgreSQL (Supabase) in production
    # For Supabase, use: postgresql+asyncpg://user:pass@host:5432/postgres
    DATABASE_URL: str = "sqlite+aiosqlite:///./deepevals.db"

    # API Keys
    ANTHROPIC_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    OPENROUTER_API_KEY: str = ""
    GROQ_API_KEY: str = ""

    class Config:
        env_file = ".env"
        extra = "ignore"


# Available models configuration
AVAILABLE_MODELS = [
    # Anthropic
    ModelConfig("anthropic", "claude-sonnet-4-5-20250929", "Claude Sonnet 4.5"),
    ModelConfig("anthropic", "claude-haiku-4-5-20251001", "Claude Haiku 4.5"),
    # OpenAI
    ModelConfig("openai", "gpt-4", "GPT-4"),
    ModelConfig("openai", "gpt-4o", "GPT-4o"),
    # Groq (fast inference)
    ModelConfig("groq", "openai/gpt-oss-120b", "GPT-OSS 120B"),
    ModelConfig("groq", "openai/gpt-oss-20b", "GPT-OSS 20B"),
    ModelConfig("groq", "qwen/qwen3-32b", "Qwen 3 32B"),
    ModelConfig("groq", "llama-3.3-70b-versatile", "Llama 3.3 70B"),
    ModelConfig("groq", "llama-3.1-8b-instant", "Llama 3.1 8B Instant"),
    ModelConfig("groq", "meta-llama/llama-4-maverick-17b-128e-instruct", "Llama 4 Maverick 17B"),
    ModelConfig("groq", "meta-llama/llama-4-scout-17b-16e-instruct", "Llama 4 Scout 17B"),
    ModelConfig("groq", "moonshotai/kimi-k2-instruct", "Kimi K2"),
    # OpenRouter
    ModelConfig("openrouter", "deepseek/deepseek-chat", "DeepSeek V3"),
    ModelConfig("openrouter", "deepseek/deepseek-r1", "DeepSeek R1 (Reasoning)"),
    ModelConfig("openrouter", "meta-llama/llama-3.3-70b-instruct", "Llama 3.3 70B"),
    ModelConfig("openrouter", "x-ai/grok-2-1212", "Grok 2"),
    ModelConfig("openrouter", "moonshotai/moonshot-v1-128k", "Kimi"),
    ModelConfig("openrouter", "qwen/qwen-2.5-72b-instruct", "Qwen 2.5 72B"),
    ModelConfig("openrouter", "google/gemini-2.0-flash-001", "Gemini 2.0 Flash"),
    ModelConfig("openrouter", "mistralai/mistral-large-2411", "Mistral Large"),
]

# Available chunk types for evaluation
CHUNK_TYPES = [
    {"id": "personality", "name": "Personality Architecture", "description": "Big Five analysis"},
    {"id": "emotional", "name": "Emotional World", "description": "Attachment, emotion regulation, conflict"},
    {"id": "values", "name": "Values & Motivation", "description": "Core values, career fit, motivation"},
    {"id": "superpowers", "name": "Superpowers & Kryptonite", "description": "Strengths and blind spots"},
    {"id": "best-fit-work", "name": "Best Fit: Work", "description": "Ideal job and jobs to avoid"},
    {"id": "best-fit-romantic", "name": "Best Fit: Romantic", "description": "Ideal partner and incompatibilities"},
    {"id": "best-fit-friends", "name": "Best Fit: Friends", "description": "Ideal friends and friction types"},
    {"id": "wellbeing-base", "name": "Wellbeing", "description": "Mental health screening summary"},
    {"id": "conclusion", "name": "Conclusion", "description": "Path forward"},
]

settings = Settings()
