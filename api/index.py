"""Vercel serverless entry point."""
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import sys
import os

app = FastAPI()

@app.get("/")
async def root():
    return {"status": "ok", "python": sys.version}

@app.get("/test-imports")
async def test_imports():
    errors = []

    try:
        import sqlalchemy
    except Exception as e:
        errors.append(f"sqlalchemy: {e}")

    try:
        import asyncpg
    except Exception as e:
        errors.append(f"asyncpg: {e}")

    try:
        import httpx
    except Exception as e:
        errors.append(f"httpx: {e}")

    try:
        import anthropic
    except Exception as e:
        errors.append(f"anthropic: {e}")

    try:
        from pathlib import Path
        root = Path(__file__).parent.parent
        sys.path.insert(0, str(root))
        from app.config import settings
    except Exception as e:
        errors.append(f"app.config: {e}")

    try:
        from app.database import Base
    except Exception as e:
        errors.append(f"app.database: {e}")

    try:
        from app.main import app as main_app
    except Exception as e:
        errors.append(f"app.main: {e}")

    if errors:
        return JSONResponse(status_code=500, content={"errors": errors})
    return {"status": "all imports ok"}

handler = app
