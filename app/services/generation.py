import asyncio
import random
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models import EvaluationSession, Generation
from app.providers import AnthropicProvider, OpenAIProvider, GroqProvider
from app.services.prompts import PromptService
from app.config import AVAILABLE_MODELS


class GenerationService:
    """Service for orchestrating parallel model generations."""

    def __init__(self):
        self.providers = {
            "anthropic": AnthropicProvider(),
            "openai": OpenAIProvider(),
            "groq": GroqProvider(),
        }

    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get all available models across providers."""
        models = []
        for model in AVAILABLE_MODELS:
            provider = self.providers.get(model.provider)
            if provider and provider.is_configured():
                models.append(model.to_dict())
        return models

    def get_model_by_id(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get a model configuration by its ID."""
        for model in AVAILABLE_MODELS:
            if model.id == model_id:
                return model.to_dict()
        return None

    async def create_session(
        self,
        db: AsyncSession,
        profile_data: Dict[str, Any],
        chunk_type: str,
        model_ids: List[str],
        evaluator_name: Optional[str] = None,
        runs_per_model: int = 1,
    ) -> EvaluationSession:
        """Create a new evaluation session with generations for each model."""
        profile_name = profile_data.get("name", "Unknown")

        # Clamp runs_per_model to 1-5
        runs_per_model = max(1, min(5, runs_per_model))

        # Create session
        session = EvaluationSession(
            profile_name=profile_name,
            profile_data=profile_data,
            chunk_type=chunk_type,
            evaluator_name=evaluator_name,
            status="pending",
        )
        db.add(session)
        await db.flush()

        # Build the prompt
        system_prompt = PromptService.load_system_prompt()
        user_prompt = PromptService.build_user_prompt(profile_data, chunk_type)
        full_prompt = f"System: {system_prompt}\n\n---\n\n{user_prompt}"

        # Create list of all generation tasks (model_id, run_number)
        all_generations = []
        for model_id in model_ids:
            for run_num in range(1, runs_per_model + 1):
                all_generations.append((model_id, run_num))

        # Randomize display order for blind evaluation
        display_orders = list(range(len(all_generations)))
        random.shuffle(display_orders)

        # Create generation records
        for i, (model_id, run_num) in enumerate(all_generations):
            model_config = self.get_model_by_id(model_id)
            if model_config:
                generation = Generation(
                    session_id=session.id,
                    provider=model_config["provider"],
                    model_id=model_config["id"],
                    model_display_name=model_config["display_name"],
                    display_order=display_orders[i],
                    run_number=run_num,
                    prompt_text=full_prompt,
                    status="pending",
                )
                db.add(generation)

        await db.commit()
        await db.refresh(session)
        return session

    async def run_generations(self, db: AsyncSession, session_id: str) -> None:
        """Run all pending generations for a session in parallel."""
        from app.database import async_session_maker

        # Get session and generations
        result = await db.execute(
            select(EvaluationSession).where(EvaluationSession.id == session_id)
        )
        session = result.scalar_one_or_none()
        if not session:
            return

        # Update session status
        session.status = "generating"
        await db.commit()

        # Get all pending generation IDs
        result = await db.execute(
            select(Generation.id).where(
                Generation.session_id == session_id,
                Generation.status == "pending"
            )
        )
        generation_ids = [row[0] for row in result.fetchall()]

        # Build prompts
        system_prompt = PromptService.load_system_prompt()
        user_prompt = PromptService.build_user_prompt(
            session.profile_data, session.chunk_type
        )

        # Run all generations in parallel - each with its own DB session
        tasks = [
            self._run_single_generation_isolated(gen_id, system_prompt, user_prompt, async_session_maker)
            for gen_id in generation_ids
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

        # Update session status
        await db.refresh(session)
        session.status = "evaluating"
        await db.commit()

    async def _run_single_generation_isolated(
        self,
        generation_id: str,
        system_prompt: str,
        user_prompt: str,
        session_maker,
    ) -> None:
        """Run a single model generation with its own DB session."""
        async with session_maker() as db:
            result = await db.execute(
                select(Generation).where(Generation.id == generation_id)
            )
            generation = result.scalar_one_or_none()
            if not generation:
                return

            print(f"Starting generation for {generation.model_display_name}")
            generation.status = "generating"
            await db.commit()

            provider = self.providers.get(generation.provider)
            if not provider:
                generation.status = "failed"
                generation.error_message = f"Unknown provider: {generation.provider}"
                await db.commit()
                return

            print(f"Calling {generation.provider} API for {generation.model_id}")
            try:
                api_result = await provider.generate(
                    model_id=generation.model_id,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
                print(f"Got result for {generation.model_display_name}: {len(api_result.output_text) if api_result.output_text else 0} chars, error={api_result.error}")

                if api_result.error:
                    generation.status = "failed"
                    generation.error_message = api_result.error
                else:
                    generation.status = "completed"
                    generation.output_text = api_result.output_text

                generation.input_tokens = api_result.input_tokens
                generation.output_tokens = api_result.output_tokens
                generation.generation_time_ms = api_result.generation_time_ms
            except Exception as e:
                print(f"Exception for {generation.model_display_name}: {e}")
                generation.status = "failed"
                generation.error_message = str(e)

            await db.commit()
