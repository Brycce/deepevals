from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from typing import List
import re

from app.database import get_db
from app.models import EvaluationSession, Generation, Rating
from app.schemas import (
    CreateSessionRequest,
    SessionSummary,
    SessionDetail,
    GenerationBlind,
    GenerationRevealed,
    ModelInfo,
    ChunkTypeInfo,
)
from app.services.generation import GenerationService
from app.services.analytics import AnalyticsService
from app.config import AVAILABLE_MODELS, CHUNK_TYPES

router = APIRouter(prefix="/api", tags=["evaluations"])
generation_service = GenerationService()


def strip_thinking_tokens(text: str) -> str:
    """Remove thinking/reasoning tokens from model output."""
    if not text:
        return text
    # Strip <think>...</think> and <thinking>...</thinking> tags (case insensitive, multiline)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Strip leading/trailing whitespace that might be left over
    return text.strip()


@router.get("/models", response_model=List[ModelInfo])
async def get_available_models():
    """Get list of available models that are properly configured."""
    return generation_service.get_available_models()


@router.get("/chunk-types", response_model=List[ChunkTypeInfo])
async def get_chunk_types():
    """Get list of available chunk types for evaluation."""
    return CHUNK_TYPES


@router.post("/evaluations", response_model=SessionDetail)
async def create_evaluation(
    request: CreateSessionRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Create a new evaluation session and start generations."""
    # Validate models
    available_model_ids = [m["id"] for m in generation_service.get_available_models()]
    for model_id in request.model_ids:
        if model_id not in available_model_ids:
            raise HTTPException(
                status_code=400,
                detail=f"Model {model_id} is not available or not configured"
            )

    # Validate chunk type
    valid_chunk_ids = [c["id"] for c in CHUNK_TYPES]
    if request.chunk_type not in valid_chunk_ids:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid chunk type: {request.chunk_type}"
        )

    # Create session
    session = await generation_service.create_session(
        db=db,
        profile_data=request.profile_data,
        chunk_type=request.chunk_type,
        model_ids=request.model_ids,
        evaluator_name=request.evaluator_name,
    )

    # Start generations in background
    background_tasks.add_task(
        run_generations_task, session.id
    )

    return SessionDetail(
        id=session.id,
        created_at=session.created_at,
        profile_name=session.profile_name,
        profile_data=session.profile_data,
        chunk_type=session.chunk_type,
        status=session.status,
        evaluator_name=session.evaluator_name,
        completed_at=session.completed_at,
    )


async def run_generations_task(session_id: str):
    """Background task to run all generations."""
    from app.database import async_session_maker

    try:
        async with async_session_maker() as db:
            await generation_service.run_generations(db, session_id)
    except Exception as e:
        print(f"Generation task error: {e}")
        import traceback
        traceback.print_exc()


@router.get("/evaluations", response_model=List[SessionSummary])
async def list_evaluations(db: AsyncSession = Depends(get_db)):
    """List all evaluation sessions."""
    result = await db.execute(
        select(EvaluationSession).order_by(EvaluationSession.created_at.desc())
    )
    sessions = result.scalars().all()

    summaries = []
    for session in sessions:
        # Count generations and ratings
        gen_result = await db.execute(
            select(Generation).where(Generation.session_id == session.id)
        )
        generations = gen_result.scalars().all()

        rating_count = sum(1 for g in generations if g.rating is not None)

        summaries.append(SessionSummary(
            id=session.id,
            created_at=session.created_at,
            profile_name=session.profile_name,
            chunk_type=session.chunk_type,
            status=session.status,
            total_generations=len(generations),
            completed_ratings=rating_count,
        ))

    return summaries


@router.get("/evaluations/{session_id}", response_model=SessionDetail)
async def get_evaluation(session_id: str, db: AsyncSession = Depends(get_db)):
    """Get details of a specific evaluation session."""
    result = await db.execute(
        select(EvaluationSession).where(EvaluationSession.id == session_id)
    )
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return SessionDetail(
        id=session.id,
        created_at=session.created_at,
        profile_name=session.profile_name,
        profile_data=session.profile_data,
        chunk_type=session.chunk_type,
        status=session.status,
        evaluator_name=session.evaluator_name,
        completed_at=session.completed_at,
    )


@router.get("/evaluations/{session_id}/status")
async def get_evaluation_status(session_id: str, db: AsyncSession = Depends(get_db)):
    """Get the current status of an evaluation session."""
    result = await db.execute(
        select(EvaluationSession).where(EvaluationSession.id == session_id)
    )
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get generation statuses
    gen_result = await db.execute(
        select(Generation).where(Generation.session_id == session_id)
    )
    generations = gen_result.scalars().all()

    return {
        "session_status": session.status,
        "generations": [
            {
                "display_label": chr(65 + g.display_order),
                "status": g.status,
            }
            for g in sorted(generations, key=lambda x: x.display_order)
        ]
    }


@router.get("/evaluations/{session_id}/generations", response_model=List[GenerationBlind])
async def get_generations_blind(session_id: str, db: AsyncSession = Depends(get_db)):
    """Get generations for blind evaluation (no model info revealed)."""
    result = await db.execute(
        select(Generation)
        .where(Generation.session_id == session_id)
        .options(selectinload(Generation.rating))
    )
    generations = result.scalars().all()

    if not generations:
        raise HTTPException(status_code=404, detail="No generations found")

    # Get prompt from first generation (all same prompt)
    prompt_text = generations[0].prompt_text if generations else None

    return [
        GenerationBlind(
            id=g.id,
            display_label=chr(65 + g.display_order),  # A, B, C...
            output_text=strip_thinking_tokens(g.output_text),
            prompt_text=prompt_text,
            status=g.status,
            has_rating=g.rating is not None,
        )
        for g in sorted(generations, key=lambda x: x.display_order)
    ]


@router.get("/evaluations/{session_id}/reveal", response_model=List[GenerationRevealed])
async def reveal_generations(session_id: str, db: AsyncSession = Depends(get_db)):
    """Reveal full generation details including model info (only after all rated)."""
    result = await db.execute(
        select(Generation)
        .where(Generation.session_id == session_id)
        .options(selectinload(Generation.rating))
    )
    generations = result.scalars().all()

    if not generations:
        raise HTTPException(status_code=404, detail="No generations found")

    # Check if all generations are rated
    completed_gens = [g for g in generations if g.status == "completed"]
    unrated = [g for g in completed_gens if g.rating is None]

    if unrated:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot reveal yet - {len(unrated)} generations still need ratings"
        )

    return [
        GenerationRevealed(
            id=g.id,
            display_label=chr(65 + g.display_order),
            provider=g.provider,
            model_id=g.model_id,
            model_display_name=g.model_display_name,
            output_text=g.output_text,
            input_tokens=g.input_tokens,
            output_tokens=g.output_tokens,
            generation_time_ms=g.generation_time_ms,
            status=g.status,
            error_message=g.error_message,
            is_good=g.rating.is_good if g.rating else None,
            rating_notes=g.rating.notes if g.rating else None,
        )
        for g in sorted(generations, key=lambda x: x.display_order)
    ]


@router.delete("/evaluations/{session_id}")
async def delete_evaluation(session_id: str, db: AsyncSession = Depends(get_db)):
    """Delete an evaluation session and all associated data."""
    result = await db.execute(
        select(EvaluationSession).where(EvaluationSession.id == session_id)
    )
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    await db.delete(session)
    await db.commit()

    return {"status": "deleted", "session_id": session_id}


@router.get("/analytics/summary")
async def get_analytics_summary(db: AsyncSession = Depends(get_db)):
    """Get aggregate analytics across all evaluations."""
    return await AnalyticsService.get_aggregate_stats(db)


@router.get("/analytics/session/{session_id}")
async def get_session_analytics(session_id: str, db: AsyncSession = Depends(get_db)):
    """Get detailed analytics for a specific session."""
    summary = await AnalyticsService.get_session_summary(db, session_id)
    if not summary:
        raise HTTPException(status_code=404, detail="Session not found")
    return summary
