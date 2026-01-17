"""Vercel serverless entry point."""
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
root_path = str(Path(__file__).parent.parent)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

# Set working directory
os.chdir(root_path)

try:
    from app.main import app
except Exception as e:
    # Create a minimal error app if imports fail
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    app = FastAPI()

    @app.get("/{path:path}")
    async def error_handler(path: str):
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "type": type(e).__name__}
        )

# Vercel expects 'app'
handler = app
