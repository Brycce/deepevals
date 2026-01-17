from typing import Dict, Any, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from collections import defaultdict

from app.models import EvaluationSession, Generation, Rating


# Cost per 1M tokens (approximate, as of Jan 2026)
COST_PER_MILLION_TOKENS = {
    # Anthropic
    "claude-sonnet-4-5-20250929": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5-20251001": {"input": 0.25, "output": 1.25},
    # OpenAI
    "gpt-4": {"input": 30.0, "output": 60.0},
    "gpt-4o": {"input": 2.50, "output": 10.0},
    # Groq (very cheap/free tier)
    "qwen-2.5-72b": {"input": 0.59, "output": 0.79},
    "qwen-2.5-32b": {"input": 0.29, "output": 0.39},
    "qwen-qwq-32b": {"input": 0.29, "output": 0.39},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "mixtral-8x7b-32768": {"input": 0.24, "output": 0.24},
    "gemma2-9b-it": {"input": 0.20, "output": 0.20},
    # OpenRouter (approximate)
    "deepseek/deepseek-chat": {"input": 0.27, "output": 1.10},
    "deepseek/deepseek-r1": {"input": 0.55, "output": 2.19},
    "meta-llama/llama-3.3-70b-instruct": {"input": 0.12, "output": 0.30},
    "x-ai/grok-2-1212": {"input": 5.0, "output": 15.0},
    "moonshotai/moonshot-v1-128k": {"input": 1.0, "output": 2.0},
    "qwen/qwen-2.5-72b-instruct": {"input": 0.35, "output": 0.40},
    "google/gemini-2.0-flash-001": {"input": 0.10, "output": 0.40},
    "mistralai/mistral-large-2411": {"input": 2.0, "output": 6.0},
}


class AnalyticsService:
    """Service for calculating evaluation analytics."""

    @staticmethod
    def estimate_cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate the cost for a generation."""
        costs = COST_PER_MILLION_TOKENS.get(model_id, {"input": 1.0, "output": 2.0})
        input_cost = (input_tokens / 1_000_000) * costs["input"]
        output_cost = (output_tokens / 1_000_000) * costs["output"]
        return round(input_cost + output_cost, 6)

    @staticmethod
    async def get_session_summary(
        db: AsyncSession, session_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get summary statistics for a single session."""
        result = await db.execute(
            select(EvaluationSession).where(EvaluationSession.id == session_id)
        )
        session = result.scalar_one_or_none()
        if not session:
            return None

        # Get generations with ratings
        result = await db.execute(
            select(Generation).where(Generation.session_id == session_id)
        )
        generations = result.scalars().all()

        summary = {
            "session_id": session.id,
            "profile_name": session.profile_name,
            "chunk_type": session.chunk_type,
            "status": session.status,
            "created_at": session.created_at.isoformat(),
            "generations": [],
            "totals": {
                "total_generations": len(generations),
                "completed_generations": 0,
                "rated_generations": 0,
                "good_ratings": 0,
                "bad_ratings": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_time_ms": 0,
                "total_cost": 0.0,
            }
        }

        for gen in sorted(generations, key=lambda g: g.display_order):
            gen_data = {
                "id": gen.id,
                "display_label": chr(65 + gen.display_order),  # A, B, C...
                "model_id": gen.model_id,
                "model_display_name": gen.model_display_name,
                "provider": gen.provider,
                "status": gen.status,
                "input_tokens": gen.input_tokens or 0,
                "output_tokens": gen.output_tokens or 0,
                "generation_time_ms": gen.generation_time_ms or 0,
                "cost": AnalyticsService.estimate_cost(
                    gen.model_id, gen.input_tokens or 0, gen.output_tokens or 0
                ),
                "is_good": None,
                "rating_notes": None,
            }

            if gen.status == "completed":
                summary["totals"]["completed_generations"] += 1

            if gen.rating:
                gen_data["is_good"] = gen.rating.is_good
                gen_data["rating_notes"] = gen.rating.notes
                summary["totals"]["rated_generations"] += 1
                if gen.rating.is_good:
                    summary["totals"]["good_ratings"] += 1
                else:
                    summary["totals"]["bad_ratings"] += 1

            summary["totals"]["total_input_tokens"] += gen_data["input_tokens"]
            summary["totals"]["total_output_tokens"] += gen_data["output_tokens"]
            summary["totals"]["total_time_ms"] += gen_data["generation_time_ms"]
            summary["totals"]["total_cost"] += gen_data["cost"]

            summary["generations"].append(gen_data)

        summary["totals"]["total_cost"] = round(summary["totals"]["total_cost"], 6)

        return summary

    @staticmethod
    async def get_aggregate_stats(db: AsyncSession) -> Dict[str, Any]:
        """Get aggregate statistics across all evaluations."""
        # Total counts
        session_count = await db.scalar(
            select(func.count(EvaluationSession.id))
        )
        generation_count = await db.scalar(
            select(func.count(Generation.id))
        )
        rating_count = await db.scalar(
            select(func.count(Rating.id))
        )

        # Per-model statistics
        result = await db.execute(
            select(
                Generation.model_id,
                Generation.model_display_name,
                func.count(Generation.id).label("total_generations"),
                func.sum(Generation.input_tokens).label("total_input_tokens"),
                func.sum(Generation.output_tokens).label("total_output_tokens"),
                func.avg(Generation.generation_time_ms).label("avg_time_ms"),
            ).group_by(Generation.model_id, Generation.model_display_name)
        )
        model_stats_raw = result.all()

        # Get ratings per model
        model_ratings = defaultdict(lambda: {"good": 0, "bad": 0, "total": 0})
        result = await db.execute(
            select(Generation.model_id, Rating.is_good)
            .join(Rating, Rating.generation_id == Generation.id)
        )
        for model_id, is_good in result.all():
            model_ratings[model_id]["total"] += 1
            if is_good:
                model_ratings[model_id]["good"] += 1
            else:
                model_ratings[model_id]["bad"] += 1

        model_stats = []
        for row in model_stats_raw:
            ratings = model_ratings[row.model_id]
            win_rate = (
                (ratings["good"] / ratings["total"] * 100)
                if ratings["total"] > 0
                else 0
            )
            model_stats.append({
                "model_id": row.model_id,
                "model_display_name": row.model_display_name,
                "total_generations": row.total_generations,
                "total_input_tokens": row.total_input_tokens or 0,
                "total_output_tokens": row.total_output_tokens or 0,
                "avg_time_ms": round(row.avg_time_ms or 0, 0),
                "total_ratings": ratings["total"],
                "good_ratings": ratings["good"],
                "bad_ratings": ratings["bad"],
                "win_rate": round(win_rate, 1),
                "estimated_cost": AnalyticsService.estimate_cost(
                    row.model_id,
                    row.total_input_tokens or 0,
                    row.total_output_tokens or 0
                ),
            })

        # Sort by win rate descending
        model_stats.sort(key=lambda x: x["win_rate"], reverse=True)

        return {
            "total_sessions": session_count or 0,
            "total_generations": generation_count or 0,
            "total_ratings": rating_count or 0,
            "model_stats": model_stats,
        }
