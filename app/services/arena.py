import asyncio
import hashlib
import json
import random
import re
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from datetime import datetime

from app.models import ArenaModel, ArenaMatch, ArenaOutput
from app.providers import AnthropicProvider, OpenAIProvider, GroqProvider
from app.services.prompts import PromptService
from app.config import AVAILABLE_MODELS


def calculate_elo(rating_a: int, rating_b: int, winner: str, k: int = 32) -> Tuple[int, int]:
    """Calculate new Elo ratings after a match."""
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
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def hash_profile(profile_data: Dict[str, Any]) -> str:
    """Create a consistent hash of profile data for cache lookup."""
    # Sort keys for consistent hashing
    json_str = json.dumps(profile_data, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()[:32]


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
            result = await db.execute(
                select(ArenaModel).where(ArenaModel.model_id == model["id"])
            )
            existing = result.scalar_one_or_none()

            if not existing:
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

    async def get_cached_output(
        self, db: AsyncSession, profile_hash: str, chunk_type: str, model_id: str
    ) -> Optional[ArenaOutput]:
        """Get cached output for a model if it exists and is completed."""
        result = await db.execute(
            select(ArenaOutput).where(
                and_(
                    ArenaOutput.profile_hash == profile_hash,
                    ArenaOutput.chunk_type == chunk_type,
                    ArenaOutput.model_id == model_id,
                    ArenaOutput.status == "completed",
                )
            )
        )
        return result.scalar_one_or_none()

    async def get_cached_outputs_for_profile(
        self, db: AsyncSession, profile_hash: str, chunk_type: str
    ) -> Dict[str, ArenaOutput]:
        """Get all cached outputs for a profile/chunk combination."""
        result = await db.execute(
            select(ArenaOutput).where(
                and_(
                    ArenaOutput.profile_hash == profile_hash,
                    ArenaOutput.chunk_type == chunk_type,
                    ArenaOutput.status == "completed",
                )
            )
        )
        outputs = result.scalars().all()
        return {o.model_id: o for o in outputs}

    async def get_models_needing_generation(
        self, db: AsyncSession, profile_hash: str, chunk_type: str
    ) -> List[str]:
        """Get model IDs that don't have cached outputs yet."""
        available_models = self.get_available_models()
        available_ids = {m["id"] for m in available_models}

        # Get models that already have outputs (completed or generating)
        result = await db.execute(
            select(ArenaOutput.model_id).where(
                and_(
                    ArenaOutput.profile_hash == profile_hash,
                    ArenaOutput.chunk_type == chunk_type,
                    ArenaOutput.status.in_(["completed", "generating"]),
                )
            )
        )
        existing_ids = {row[0] for row in result.fetchall()}

        return list(available_ids - existing_ids)

    async def select_pair_for_match(
        self, db: AsyncSession, profile_hash: str, chunk_type: str,
        exclude_pairs: Optional[List[Tuple[str, str]]] = None
    ) -> Tuple[str, str]:
        """Select two models for a match, preferring those with cached outputs."""
        available_models = self.get_available_models()
        if len(available_models) < 2:
            raise ValueError("Need at least 2 models for arena matchup")

        # Get cached outputs
        cached = await self.get_cached_outputs_for_profile(db, profile_hash, chunk_type)
        cached_ids = set(cached.keys())
        all_ids = [m["id"] for m in available_models]

        # Get previously played pairs for this profile
        if exclude_pairs is None:
            result = await db.execute(
                select(ArenaMatch.model_a_id, ArenaMatch.model_b_id).where(
                    and_(
                        ArenaMatch.profile_hash == profile_hash,
                        ArenaMatch.chunk_type == chunk_type,
                    )
                )
            )
            played_pairs = {tuple(sorted([row[0], row[1]])) for row in result.fetchall()}
        else:
            played_pairs = {tuple(sorted(p)) for p in exclude_pairs}

        # Try to find a pair we haven't played yet, preferring cached models
        cached_list = [m for m in all_ids if m in cached_ids]
        uncached_list = [m for m in all_ids if m not in cached_ids]

        # Strategy: prefer pairs where at least one model is cached
        # First priority: both cached, not played
        # Second priority: one cached + one uncached, not played
        # Third priority: any unplayed pair
        # Last resort: random pair (even if played before)

        def find_unplayed_pair(candidates: List[str]) -> Optional[Tuple[str, str]]:
            random.shuffle(candidates)
            for i, a in enumerate(candidates):
                for b in candidates[i+1:]:
                    if tuple(sorted([a, b])) not in played_pairs:
                        return (a, b)
            return None

        # Try both cached
        if len(cached_list) >= 2:
            pair = find_unplayed_pair(cached_list)
            if pair:
                return pair

        # Try one cached + one uncached
        if cached_list and uncached_list:
            random.shuffle(cached_list)
            random.shuffle(uncached_list)
            for a in cached_list:
                for b in uncached_list:
                    if tuple(sorted([a, b])) not in played_pairs:
                        return (a, b)

        # Try any unplayed pair
        pair = find_unplayed_pair(all_ids)
        if pair:
            return pair

        # Last resort: random pair
        selected = random.sample(all_ids, 2)
        return (selected[0], selected[1])

    async def create_match(
        self,
        db: AsyncSession,
        profile_data: Dict[str, Any],
        chunk_type: str,
    ) -> ArenaMatch:
        """Create a new arena match, reusing cached outputs when available."""
        profile_hash = hash_profile(profile_data)
        profile_name = profile_data.get("name", "Unknown")

        # Select models (prefers those with cached outputs)
        model_a_id, model_b_id = await self.select_pair_for_match(db, profile_hash, chunk_type)

        # Check for cached outputs
        cached_a = await self.get_cached_output(db, profile_hash, chunk_type, model_a_id)
        cached_b = await self.get_cached_output(db, profile_hash, chunk_type, model_b_id)

        # Create match record
        match = ArenaMatch(
            profile_name=profile_name,
            profile_data=profile_data,
            profile_hash=profile_hash,
            chunk_type=chunk_type,
            model_a_id=model_a_id,
            model_b_id=model_b_id,
            model_a_output=cached_a.output_text if cached_a else None,
            model_b_output=cached_b.output_text if cached_b else None,
            status="ready" if (cached_a and cached_b) else "pending",
        )
        db.add(match)
        await db.commit()
        await db.refresh(match)

        return match

    async def generate_single_output(
        self,
        profile_data: Dict[str, Any],
        profile_hash: str,
        chunk_type: str,
        model_id: str,
    ) -> Optional[str]:
        """Generate output for a single model and cache it."""
        from app.database import async_session_maker

        model_config = self.get_model_by_id(model_id)
        if not model_config:
            return None

        provider = self.providers.get(model_config["provider"])
        if not provider:
            return None

        async with async_session_maker() as db:
            # Check if already exists or being generated
            result = await db.execute(
                select(ArenaOutput).where(
                    and_(
                        ArenaOutput.profile_hash == profile_hash,
                        ArenaOutput.chunk_type == chunk_type,
                        ArenaOutput.model_id == model_id,
                    )
                )
            )
            existing = result.scalar_one_or_none()

            if existing:
                if existing.status == "completed":
                    return existing.output_text
                elif existing.status == "generating":
                    return None  # Already being generated

            # Create or update output record
            if not existing:
                output_record = ArenaOutput(
                    profile_hash=profile_hash,
                    chunk_type=chunk_type,
                    model_id=model_id,
                    status="generating",
                )
                db.add(output_record)
            else:
                output_record = existing
                output_record.status = "generating"

            await db.commit()
            await db.refresh(output_record)
            output_id = output_record.id

        # Generate outside the session
        system_prompt = PromptService.load_system_prompt()
        user_prompt = PromptService.build_user_prompt(profile_data, chunk_type)
        max_tokens = 16000 if chunk_type == "full-report" else 4000

        try:
            api_result = await provider.generate(
                model_id=model_id,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
            )

            async with async_session_maker() as db:
                result = await db.execute(
                    select(ArenaOutput).where(ArenaOutput.id == output_id)
                )
                output_record = result.scalar_one_or_none()
                if output_record:
                    if api_result.error:
                        output_record.status = "failed"
                    else:
                        output_record.output_text = api_result.output_text
                        output_record.status = "completed"
                    await db.commit()
                    return output_record.output_text if not api_result.error else None

        except Exception as e:
            print(f"Arena generation error for {model_id}: {e}")
            async with async_session_maker() as db:
                result = await db.execute(
                    select(ArenaOutput).where(ArenaOutput.id == output_id)
                )
                output_record = result.scalar_one_or_none()
                if output_record:
                    output_record.status = "failed"
                    await db.commit()

        return None

    async def run_match_generations(self, db: AsyncSession, match_id: str) -> None:
        """Run generations for models that don't have cached outputs."""
        from app.database import async_session_maker

        result = await db.execute(
            select(ArenaMatch).where(ArenaMatch.id == match_id)
        )
        match = result.scalar_one_or_none()
        if not match:
            return

        # If already ready, nothing to do
        if match.status == "ready":
            return

        match.status = "generating"
        await db.commit()

        profile_hash = match.profile_hash or hash_profile(match.profile_data)
        tasks = []

        # Only generate for models that need it
        if not match.model_a_output:
            tasks.append(self.generate_single_output(
                match.profile_data, profile_hash, match.chunk_type, match.model_a_id
            ))
        if not match.model_b_output:
            tasks.append(self.generate_single_output(
                match.profile_data, profile_hash, match.chunk_type, match.model_b_id
            ))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Update match with results
        async with async_session_maker() as session:
            result = await session.execute(
                select(ArenaMatch).where(ArenaMatch.id == match_id)
            )
            m = result.scalar_one_or_none()
            if m:
                # Get cached outputs
                cached_a = await self.get_cached_output(session, profile_hash, match.chunk_type, m.model_a_id)
                cached_b = await self.get_cached_output(session, profile_hash, match.chunk_type, m.model_b_id)

                if cached_a:
                    m.model_a_output = cached_a.output_text
                if cached_b:
                    m.model_b_output = cached_b.output_text

                m.status = "ready"
                await session.commit()

    async def pregenerate_remaining_models(
        self, profile_data: Dict[str, Any], chunk_type: str
    ) -> None:
        """Background task to pre-generate outputs for remaining models."""
        from app.database import async_session_maker

        profile_hash = hash_profile(profile_data)

        async with async_session_maker() as db:
            models_needed = await self.get_models_needing_generation(db, profile_hash, chunk_type)

        if not models_needed:
            return

        # Generate in parallel (limit concurrency to avoid overwhelming APIs)
        semaphore = asyncio.Semaphore(3)

        async def generate_with_limit(model_id: str):
            async with semaphore:
                await self.generate_single_output(profile_data, profile_hash, chunk_type, model_id)

        await asyncio.gather(*[generate_with_limit(m) for m in models_needed], return_exceptions=True)

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
        """Submit a vote for a match and update Elo ratings."""
        if winner not in ("a", "b", "tie"):
            raise ValueError("Winner must be 'a', 'b', or 'tie'")

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
