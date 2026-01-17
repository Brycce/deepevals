from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime


# Request schemas
class CreateSessionRequest(BaseModel):
    profile_data: Dict[str, Any]
    chunk_type: str
    model_ids: List[str]  # List of model IDs to use
    evaluator_name: Optional[str] = None


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
