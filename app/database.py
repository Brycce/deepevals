from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from app.config import settings
import os


class Base(DeclarativeBase):
    pass


def get_database_url():
    """Convert database URL to async driver format."""
    url = settings.DATABASE_URL
    print(f"[DEBUG] Original DATABASE_URL: {url[:50]}..." if len(url) > 50 else f"[DEBUG] Original DATABASE_URL: {url}")

    # Convert postgresql:// to postgresql+asyncpg:// for async support
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)

    print(f"[DEBUG] Final database URL scheme: {url.split('://')[0]}")
    return url


db_url = get_database_url()
engine = create_async_engine(db_url, echo=False)
async_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db():
    """Create all database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    """Dependency for getting database sessions."""
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()
