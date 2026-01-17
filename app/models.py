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
