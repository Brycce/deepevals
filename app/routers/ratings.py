from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from datetime import datetime

from app.database import get_db
from app.models import Generation, Rating, EvaluationSession
from app.schemas import SubmitRatingRequest

router = APIRouter(prefix="/api", tags=["ratings"])


@router.post("/ratings")
async def submit_rating(
    request: SubmitRatingRequest,
    db: AsyncSession = Depends(get_db),
):
    """Submit a rating for a generation."""
    # Get the generation with rating eagerly loaded
    result = await db.execute(
        select(Generation)
        .where(Generation.id == request.generation_id)
        .options(selectinload(Generation.rating))
    )
    generation = result.scalar_one_or_none()

    if not generation:
        raise HTTPException(status_code=404, detail="Generation not found")

    if generation.status != "completed":
        raise HTTPException(
            status_code=400,
            detail="Cannot rate a generation that hasn't completed"
        )

    # Check if already rated
    if generation.rating:
        # Update existing rating
        generation.rating.is_good = request.is_good
        generation.rating.notes = request.notes
        generation.rating.created_at = datetime.utcnow()
    else:
        # Create new rating
        rating = Rating(
            generation_id=generation.id,
            is_good=request.is_good,
            notes=request.notes,
        )
        db.add(rating)

    await db.commit()

    # Check if all generations in the session are now rated
    await _check_session_completion(db, generation.session_id)

    return {
        "status": "success",
        "generation_id": generation.id,
        "is_good": request.is_good,
    }


@router.get("/ratings/{generation_id}")
async def get_rating(generation_id: str, db: AsyncSession = Depends(get_db)):
    """Get the rating for a specific generation."""
    result = await db.execute(
        select(Generation)
        .where(Generation.id == generation_id)
        .options(selectinload(Generation.rating))
    )
    generation = result.scalar_one_or_none()

    if not generation:
        raise HTTPException(status_code=404, detail="Generation not found")

    if not generation.rating:
        return {"has_rating": False}

    return {
        "has_rating": True,
        "is_good": generation.rating.is_good,
        "notes": generation.rating.notes,
        "created_at": generation.rating.created_at.isoformat(),
    }


@router.delete("/ratings/{generation_id}")
async def delete_rating(generation_id: str, db: AsyncSession = Depends(get_db)):
    """Delete a rating for a generation."""
    result = await db.execute(
        select(Generation)
        .where(Generation.id == generation_id)
        .options(selectinload(Generation.rating))
    )
    generation = result.scalar_one_or_none()

    if not generation:
        raise HTTPException(status_code=404, detail="Generation not found")

    if not generation.rating:
        raise HTTPException(status_code=404, detail="No rating found")

    await db.delete(generation.rating)
    await db.commit()

    # Update session status if needed
    session_result = await db.execute(
        select(EvaluationSession).where(EvaluationSession.id == generation.session_id)
    )
    session = session_result.scalar_one_or_none()
    if session and session.status == "completed":
        session.status = "evaluating"
        session.completed_at = None
        await db.commit()

    return {"status": "deleted", "generation_id": generation_id}


async def _check_session_completion(db: AsyncSession, session_id: str) -> None:
    """Check if all generations in a session are rated and update status."""
    # Get all completed generations with ratings eagerly loaded
    result = await db.execute(
        select(Generation)
        .where(
            Generation.session_id == session_id,
            Generation.status == "completed"
        )
        .options(selectinload(Generation.rating))
    )
    generations = result.scalars().all()

    # Check if all have ratings
    all_rated = all(g.rating is not None for g in generations)

    if all_rated and generations:
        # Update session to completed
        session_result = await db.execute(
            select(EvaluationSession).where(EvaluationSession.id == session_id)
        )
        session = session_result.scalar_one_or_none()
        if session:
            session.status = "completed"
            session.completed_at = datetime.utcnow()
            await db.commit()
