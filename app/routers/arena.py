from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from app.database import get_db
from app.schemas import (
    ArenaMatchCreate,
    ArenaMatchBlind,
    ArenaMatchRevealed,
    ArenaVoteRequest,
    ArenaLeaderboardEntry,
    ArenaLeaderboardResponse,
)
from app.services.arena import ArenaService, strip_thinking_tokens
from app.config import CHUNK_TYPES

router = APIRouter(prefix="/api/arena", tags=["arena"])
arena_service = ArenaService()


@router.get("/leaderboard", response_model=ArenaLeaderboardResponse)
async def get_leaderboard(db: AsyncSession = Depends(get_db)):
    """Get current Elo rankings for all models."""
    entries = await arena_service.get_leaderboard(db)
    return ArenaLeaderboardResponse(
        entries=[ArenaLeaderboardEntry(**e) for e in entries]
    )


@router.post("/match", response_model=ArenaMatchBlind)
async def create_match(
    request: ArenaMatchCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Create a new arena match with randomly selected models."""
    # Validate chunk type
    valid_chunk_ids = [c["id"] for c in CHUNK_TYPES]
    if request.chunk_type not in valid_chunk_ids:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid chunk type: {request.chunk_type}"
        )

    # Check we have at least 2 models available
    available_models = arena_service.get_available_models()
    if len(available_models) < 2:
        raise HTTPException(
            status_code=400,
            detail="Need at least 2 models configured for arena mode"
        )

    # Create match
    match = await arena_service.create_match(
        db=db,
        profile_data=request.profile_data,
        chunk_type=request.chunk_type,
    )

    # Start generations in background (if not already ready from cache)
    if match.status != "ready":
        background_tasks.add_task(
            run_match_generations_task,
            match.id,
            request.profile_data,
            request.chunk_type
        )
    else:
        # Match already ready, just pre-generate remaining models
        background_tasks.add_task(
            arena_service.pregenerate_remaining_models,
            request.profile_data,
            request.chunk_type
        )

    return ArenaMatchBlind(
        id=match.id,
        created_at=match.created_at,
        profile_name=match.profile_name,
        chunk_type=match.chunk_type,
        status=match.status,
        output_a=None,
        output_b=None,
    )


async def run_match_generations_task(match_id: str, profile_data: dict = None, chunk_type: str = None):
    """Background task to run match generations and pre-generate remaining models."""
    from app.database import async_session_maker

    try:
        async with async_session_maker() as db:
            await arena_service.run_match_generations(db, match_id)

        # After match is ready, pre-generate outputs for remaining models in background
        if profile_data and chunk_type:
            await arena_service.pregenerate_remaining_models(profile_data, chunk_type)
    except Exception as e:
        print(f"Arena match generation error: {e}")
        import traceback
        traceback.print_exc()


@router.get("/match/{match_id}", response_model=ArenaMatchBlind)
async def get_match_blind(match_id: str, db: AsyncSession = Depends(get_db)):
    """Get match details without revealing model identities."""
    match = await arena_service.get_match(db, match_id)
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")

    return ArenaMatchBlind(
        id=match.id,
        created_at=match.created_at,
        profile_name=match.profile_name,
        chunk_type=match.chunk_type,
        status=match.status,
        output_a=strip_thinking_tokens(match.model_a_output),
        output_b=strip_thinking_tokens(match.model_b_output),
    )


@router.get("/match/{match_id}/status")
async def get_match_status(match_id: str, db: AsyncSession = Depends(get_db)):
    """Get the current status of a match."""
    match = await arena_service.get_match(db, match_id)
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")

    return {
        "status": match.status,
        "has_output_a": match.model_a_output is not None,
        "has_output_b": match.model_b_output is not None,
    }


@router.post("/match/{match_id}/vote")
async def submit_vote(
    match_id: str,
    request: ArenaVoteRequest,
    db: AsyncSession = Depends(get_db),
):
    """Submit a vote for the match winner."""
    if request.winner not in ("a", "b", "tie"):
        raise HTTPException(
            status_code=400,
            detail="Winner must be 'a', 'b', or 'tie'"
        )

    try:
        result = await arena_service.submit_vote(db, match_id, request.winner)
        if not result:
            raise HTTPException(status_code=404, detail="Match not found")
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/match/{match_id}/full")
async def get_match_full(match_id: str, db: AsyncSession = Depends(get_db)):
    """Get full match data including profile_data (for creating new matches)."""
    match = await arena_service.get_match(db, match_id)
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")

    return {
        "id": match.id,
        "profile_name": match.profile_name,
        "profile_data": match.profile_data,
        "chunk_type": match.chunk_type,
    }


@router.get("/match/{match_id}/reveal", response_model=ArenaMatchRevealed)
async def reveal_match(match_id: str, db: AsyncSession = Depends(get_db)):
    """Reveal model identities after voting."""
    match = await arena_service.get_match(db, match_id)
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")

    if match.status != "completed":
        raise HTTPException(
            status_code=400,
            detail="Cannot reveal until match is completed (vote submitted)"
        )

    # Get display names
    model_a_config = arena_service.get_model_by_id(match.model_a_id)
    model_b_config = arena_service.get_model_by_id(match.model_b_id)

    return ArenaMatchRevealed(
        id=match.id,
        created_at=match.created_at,
        profile_name=match.profile_name,
        chunk_type=match.chunk_type,
        status=match.status,
        output_a=match.model_a_output,
        output_b=match.model_b_output,
        model_a_id=match.model_a_id,
        model_a_display_name=model_a_config["display_name"] if model_a_config else match.model_a_id,
        model_b_id=match.model_b_id,
        model_b_display_name=model_b_config["display_name"] if model_b_config else match.model_b_id,
        winner=match.winner,
        completed_at=match.completed_at,
    )
