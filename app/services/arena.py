import asyncio
import random
import re
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime

from app.models import ArenaModel, ArenaMatch
from app.providers import AnthropicProvider, OpenAIProvider, GroqProvider
from app.services.prompts import PromptService
from app.config import AVAILABLE_MODELS


def calculate_elo(rating_a: int, rating_b: int, winner: str, k: int = 32) -> Tuple[int, int]:
    """Calculate new Elo ratings after a match.

    Args:
        rating_a: Current Elo rating of model A
        rating_b: Current Elo rating of model B
        winner: "a", "b", or "tie"
        k: K-factor (default 32)

    Returns:
        Tuple of (new_rating_a, new_rating_b)
    """
    expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    expected_b = 1 - expected_a

    if winner == "a":
        score_a, score_b = 1, 0
    elif winner == "b":
        score_a, score_b = 0, 1
    else:  # tie
        score_a, score_b = 0.5, 0.5

    new_rating_a = round(rating_a + k * (score_a - expected_a))
    new_rating_b = round(rating_b + k * (score_b - expected_b))

    return new_rating_a, new_rating_b


def strip_thinking_tokens(text: str) -> str:
    """Remove thinking/reasoning tokens from model output."""
    if not text:
        return text
    # Strip <think>...</think> and <thinking>...</thinking> tags
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


class ArenaService:
    """Service for Arena Mode matchups and Elo rating calculations."""

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

    async def ensure_arena_models_exist(self, db: AsyncSession) -> None:
        """Ensure all available models have ArenaModel records."""
        available_models = self.get_available_models()

        for model in available_models:
            # Check if model exists
            result = await db.execute(
                select(ArenaModel).where(ArenaModel.model_id == model["id"])
            )
            existing = result.scalar_one_or_none()

            if not existing:
                # Create new arena model record
                arena_model = ArenaModel(
                    model_id=model["id"],
                    model_display_name=model["display_name"],
                    elo_rating=1500,
                    matches_played=0,
                    wins=0,
                    losses=0,
                    ties=0,
                )
                db.add(arena_model)

        await db.commit()

    async def get_leaderboard(self, db: AsyncSession) -> List[Dict[str, Any]]:
        """Get the current leaderboard sorted by Elo rating."""
        # Ensure all models exist first
        await self.ensure_arena_models_exist(db)

        result = await db.execute(
            select(ArenaModel).order_by(ArenaModel.elo_rating.desc())
        )
        models = result.scalars().all()

        return [
            {
                "model_id": m.model_id,
                "model_display_name": m.model_display_name,
                "elo_rating": m.elo_rating,
                "matches_played": m.matches_played,
                "wins": m.wins,
                "losses": m.losses,
                "ties": m.ties,
            }
            for m in models
        ]

    def select_random_pair(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Select two random models for a matchup."""
        available_models = self.get_available_models()

        if len(available_models) < 2:
            raise ValueError("Need at least 2 models for arena matchup")

        # Randomly select 2 different models
        selected = random.sample(available_models, 2)
        return selected[0], selected[1]

    async def create_match(
        self,
        db: AsyncSession,
        profile_data: Dict[str, Any],
        chunk_type: str,
    ) -> ArenaMatch:
        """Create a new arena match with randomly selected models."""
        # Select random model pair
        model_a, model_b = self.select_random_pair()
        profile_name = profile_data.get("name", "Unknown")

        # Create match record
        match = ArenaMatch(
            profile_name=profile_name,
            profile_data=profile_data,
            chunk_type=chunk_type,
            model_a_id=model_a["id"],
            model_b_id=model_b["id"],
            status="pending",
        )
        db.add(match)
        await db.commit()
        await db.refresh(match)

        return match

    async def run_match_generations(self, db: AsyncSession, match_id: str) -> None:
        """Run generations for both models in a match in parallel."""
        from app.database import async_session_maker

        # Get match
        result = await db.execute(
            select(ArenaMatch).where(ArenaMatch.id == match_id)
        )
        match = result.scalar_one_or_none()
        if not match:
            return

        # Update status
        match.status = "generating"
        await db.commit()

        # Build prompts
        system_prompt = PromptService.load_system_prompt()
        user_prompt = PromptService.build_user_prompt(
            match.profile_data, match.chunk_type
        )

        # Use more tokens for full report
        max_tokens = 16000 if match.chunk_type == "full-report" else 4000

        # Run both generations in parallel with isolated sessions
        async def generate_for_model(model_id: str, is_model_a: bool):
            async with async_session_maker() as session:
                # Get match again in this session
                result = await session.execute(
                    select(ArenaMatch).where(ArenaMatch.id == match_id)
                )
                m = result.scalar_one_or_none()
                if not m:
                    return

                model_config = self.get_model_by_id(model_id)
                if not model_config:
                    return

                provider = self.providers.get(model_config["provider"])
                if not provider:
                    return

                try:
                    api_result = await provider.generate(
                        model_id=model_id,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        max_tokens=max_tokens,
                    )

                    if not api_result.error:
                        if is_model_a:
                            m.model_a_output = api_result.output_text
                        else:
                            m.model_b_output = api_result.output_text
                        await session.commit()
                except Exception as e:
                    print(f"Arena generation error for {model_id}: {e}")

        # Run both generations in parallel
        await asyncio.gather(
            generate_for_model(match.model_a_id, True),
            generate_for_model(match.model_b_id, False),
        )

        # Update match status to ready
        await db.refresh(match)
        match.status = "ready"
        await db.commit()

    async def get_match(self, db: AsyncSession, match_id: str) -> Optional[ArenaMatch]:
        """Get a match by ID."""
        result = await db.execute(
            select(ArenaMatch).where(ArenaMatch.id == match_id)
        )
        return result.scalar_one_or_none()

    async def submit_vote(
        self,
        db: AsyncSession,
        match_id: str,
        winner: str
    ) -> Optional[Dict[str, Any]]:
        """Submit a vote for a match and update Elo ratings.

        Args:
            db: Database session
            match_id: Match ID
            winner: "a", "b", or "tie"

        Returns:
            Dict with updated ratings if successful, None if match not found
        """
        if winner not in ("a", "b", "tie"):
            raise ValueError("Winner must be 'a', 'b', or 'tie'")

        # Get match
        result = await db.execute(
            select(ArenaMatch).where(ArenaMatch.id == match_id)
        )
        match = result.scalar_one_or_none()
        if not match:
            return None

        if match.status == "completed":
            raise ValueError("Match already completed")

        if match.status != "ready":
            raise ValueError("Match not ready for voting")

        # Get arena model records
        result_a = await db.execute(
            select(ArenaModel).where(ArenaModel.model_id == match.model_a_id)
        )
        model_a = result_a.scalar_one_or_none()

        result_b = await db.execute(
            select(ArenaModel).where(ArenaModel.model_id == match.model_b_id)
        )
        model_b = result_b.scalar_one_or_none()

        # Create arena model records if they don't exist
        if not model_a:
            model_a_config = self.get_model_by_id(match.model_a_id)
            model_a = ArenaModel(
                model_id=match.model_a_id,
                model_display_name=model_a_config["display_name"] if model_a_config else match.model_a_id,
                elo_rating=1500,
            )
            db.add(model_a)

        if not model_b:
            model_b_config = self.get_model_by_id(match.model_b_id)
            model_b = ArenaModel(
                model_id=match.model_b_id,
                model_display_name=model_b_config["display_name"] if model_b_config else match.model_b_id,
                elo_rating=1500,
            )
            db.add(model_b)

        # Calculate new Elo ratings
        old_rating_a = model_a.elo_rating
        old_rating_b = model_b.elo_rating
        new_rating_a, new_rating_b = calculate_elo(old_rating_a, old_rating_b, winner)

        # Update model records
        model_a.elo_rating = new_rating_a
        model_a.matches_played += 1
        model_b.elo_rating = new_rating_b
        model_b.matches_played += 1

        if winner == "a":
            model_a.wins += 1
            model_b.losses += 1
        elif winner == "b":
            model_b.wins += 1
            model_a.losses += 1
        else:  # tie
            model_a.ties += 1
            model_b.ties += 1

        # Update match
        match.winner = winner
        match.status = "completed"
        match.completed_at = datetime.utcnow()

        await db.commit()

        return {
            "model_a": {
                "model_id": model_a.model_id,
                "model_display_name": model_a.model_display_name,
                "old_elo": old_rating_a,
                "new_elo": new_rating_a,
                "change": new_rating_a - old_rating_a,
            },
            "model_b": {
                "model_id": model_b.model_id,
                "model_display_name": model_b.model_display_name,
                "old_elo": old_rating_b,
                "new_elo": new_rating_b,
                "change": new_rating_b - old_rating_b,
            },
            "winner": winner,
        }
