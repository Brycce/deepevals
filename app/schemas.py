from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime


# Request schemas
class CreateSessionRequest(BaseModel):
    profile_data: Dict[str, Any]
    chunk_type: str
    model_ids: List[str]  # List of model IDs to use
    evaluator_name: Optional[str] = None
    runs_per_model: int = 1  # How many times each model should generate (1-5)


class SubmitRatingRequest(BaseModel):
    generation_id: str
    is_good: bool
    notes: Optional[str] = None


# Response schemas
class ModelInfo(BaseModel):
    provider: str
    id: str
    display_name: str


class ChunkTypeInfo(BaseModel):
    id: str
    name: str
    description: str


class GenerationBlind(BaseModel):
    """Generation info without revealing the model (for blind evaluation)."""
    id: str
    display_label: str  # A, B, C, etc.
    output_text: Optional[str]
    prompt_text: Optional[str]
    status: str
    has_rating: bool
    run_number: int = 1


class GenerationRevealed(BaseModel):
    """Full generation info including model details (after evaluation)."""
    id: str
    display_label: str
    provider: str
    model_id: str
    model_display_name: str
    output_text: Optional[str]
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    generation_time_ms: Optional[int]
    status: str
    error_message: Optional[str]
    is_good: Optional[bool]
    rating_notes: Optional[str]
    run_number: int = 1


class SessionSummary(BaseModel):
    id: str
    created_at: datetime
    profile_name: str
    chunk_type: str
    status: str
    total_generations: int
    completed_ratings: int


class SessionDetail(BaseModel):
    id: str
    created_at: datetime
    profile_name: str
    profile_data: Dict[str, Any]
    chunk_type: str
    status: str
    evaluator_name: Optional[str]
    completed_at: Optional[datetime]


class AnalyticsSummary(BaseModel):
    total_sessions: int
    total_generations: int
    total_ratings: int
    model_stats: List[Dict[str, Any]]  # Per-model statistics


# Arena Mode schemas
class ArenaMatchCreate(BaseModel):
    """Request to start a new arena match."""
    profile_data: Dict[str, Any]
    chunk_type: str


class ArenaMatchBlind(BaseModel):
    """Match details for display (model identities hidden)."""
    id: str
    created_at: datetime
    profile_name: str
    chunk_type: str
    status: str
    output_a: Optional[str]  # Model A's output
    output_b: Optional[str]  # Model B's output


class ArenaMatchRevealed(BaseModel):
    """Match details with model identities revealed."""
    id: str
    created_at: datetime
    profile_name: str
    chunk_type: str
    status: str
    output_a: Optional[str]
    output_b: Optional[str]
    model_a_id: str
    model_a_display_name: str
    model_b_id: str
    model_b_display_name: str
    winner: Optional[str]
    completed_at: Optional[datetime]


class ArenaVoteRequest(BaseModel):
    """Submit a vote for a match winner."""
    winner: str  # "a", "b", or "tie"


class ArenaLeaderboardEntry(BaseModel):
    """Single entry in the leaderboard."""
    model_id: str
    model_display_name: str
    elo_rating: int
    matches_played: int
    wins: int
    losses: int
    ties: int


class ArenaLeaderboardResponse(BaseModel):
    """Leaderboard data response."""
    entries: List[ArenaLeaderboardEntry]
