from sqlalchemy import Column, String, Integer, Boolean, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from app.database import Base


def generate_uuid():
    return str(uuid.uuid4())


class EvaluationSession(Base):
    """An evaluation session comparing multiple models on the same prompt."""
    __tablename__ = "evaluation_sessions"

    id = Column(String, primary_key=True, default=generate_uuid)
    created_at = Column(DateTime, default=datetime.utcnow)
    profile_name = Column(String(255), nullable=False)
    profile_data = Column(JSON, nullable=False)
    chunk_type = Column(String(100), nullable=False)
    status = Column(String(50), default="pending")  # pending, generating, evaluating, completed
    evaluator_name = Column(String(255), nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    generations = relationship("Generation", back_populates="session", cascade="all, delete-orphan")


class Generation(Base):
    """A single model generation within an evaluation session."""
    __tablename__ = "generations"

    id = Column(String, primary_key=True, default=generate_uuid)
    session_id = Column(String, ForeignKey("evaluation_sessions.id", ondelete="CASCADE"), nullable=False)
    provider = Column(String(50), nullable=False)  # anthropic, openai, openrouter
    model_id = Column(String(100), nullable=False)
    model_display_name = Column(String(100), nullable=False)
    display_order = Column(Integer, nullable=False)  # Randomized order for blind evaluation
    run_number = Column(Integer, default=1)  # Which run this is (1-5) for multiple runs per model
    prompt_text = Column(Text, nullable=False)
    output_text = Column(Text, nullable=True)
    input_tokens = Column(Integer, nullable=True)
    output_tokens = Column(Integer, nullable=True)
    generation_time_ms = Column(Integer, nullable=True)
    status = Column(String(50), default="pending")  # pending, generating, completed, failed
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    session = relationship("EvaluationSession", back_populates="generations")
    rating = relationship("Rating", back_populates="generation", uselist=False, cascade="all, delete-orphan")


class Rating(Base):
    """Human rating for a generation."""
    __tablename__ = "ratings"

    id = Column(String, primary_key=True, default=generate_uuid)
    generation_id = Column(String, ForeignKey("generations.id", ondelete="CASCADE"), nullable=False, unique=True)
    is_good = Column(Boolean, nullable=False)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    generation = relationship("Generation", back_populates="rating")


class ArenaModel(Base):
    """Tracks Elo ratings for each model in Arena Mode."""
    __tablename__ = "arena_models"

    id = Column(String, primary_key=True, default=generate_uuid)
    model_id = Column(String(100), nullable=False, unique=True)  # e.g., "claude-sonnet-4-5-20250929"
    model_display_name = Column(String(100), nullable=False)  # e.g., "Claude Sonnet 4.5"
    elo_rating = Column(Integer, default=1500)
    matches_played = Column(Integer, default=0)
    wins = Column(Integer, default=0)
    losses = Column(Integer, default=0)
    ties = Column(Integer, default=0)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ArenaMatch(Base):
    """Individual head-to-head matches in Arena Mode."""
    __tablename__ = "arena_matches"

    id = Column(String, primary_key=True, default=generate_uuid)
    created_at = Column(DateTime, default=datetime.utcnow)
    profile_name = Column(String(255), nullable=False)
    profile_data = Column(JSON, nullable=False)
    profile_hash = Column(String(64), nullable=True)  # Hash for cache lookup
    chunk_type = Column(String(100), nullable=False)
    model_a_id = Column(String(100), nullable=False)  # First model ID
    model_a_output = Column(Text, nullable=True)  # First model's output
    model_b_id = Column(String(100), nullable=False)  # Second model ID
    model_b_output = Column(Text, nullable=True)  # Second model's output
    winner = Column(String(10), nullable=True)  # "a", "b", or "tie" (null if not judged)
    status = Column(String(50), default="pending")  # pending, generating, ready, completed
    completed_at = Column(DateTime, nullable=True)


class ArenaOutput(Base):
    """Cached model outputs for arena mode - reused across matches."""
    __tablename__ = "arena_outputs"

    id = Column(String, primary_key=True, default=generate_uuid)
    profile_hash = Column(String(64), nullable=False)  # Hash of profile_data
    chunk_type = Column(String(100), nullable=False)
    model_id = Column(String(100), nullable=False)
    output_text = Column(Text, nullable=True)
    status = Column(String(50), default="pending")  # pending, generating, completed, failed
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        # Unique constraint: one output per profile+chunk+model
        {"sqlite_autoincrement": True},
    )
