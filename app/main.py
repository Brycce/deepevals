from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pathlib import Path
import markdown
import os

from app.database import init_db
from app.routers import evaluations_router, ratings_router
from app.services.generation import GenerationService
from app.config import CHUNK_TYPES

# Track if DB is initialized (for serverless)
_db_initialized = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup."""
    global _db_initialized
    if not _db_initialized:
        await init_db()
        _db_initialized = True
    yield


app = FastAPI(
    title="DeepEvals",
    description="Model evaluation tool for comparing AI-generated personality analyses",
    version="1.0.0",
    lifespan=lifespan,
)

# Mount static files
static_path = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_path), name="static")

# Mount sample profiles
sample_profiles_path = Path(__file__).parent.parent / "sample_profiles"
app.mount("/sample_profiles", StaticFiles(directory=sample_profiles_path), name="sample_profiles")

# Templates
templates_path = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=templates_path)

# Include API routers
app.include_router(evaluations_router)
app.include_router(ratings_router)

# Services
generation_service = GenerationService()


def render_markdown(text: str) -> str:
    """Convert markdown to HTML."""
    if not text:
        return ""
    return markdown.markdown(text, extensions=["tables", "fenced_code"])


# Add markdown filter to templates
templates.env.filters["markdown"] = render_markdown


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Home page - create new evaluation."""
    models = generation_service.get_available_models()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "models": models,
            "chunk_types": CHUNK_TYPES,
        }
    )


@app.get("/evaluate/{session_id}", response_class=HTMLResponse)
async def evaluate_page(request: Request, session_id: str):
    """Evaluation page for rating generations."""
    return templates.TemplateResponse(
        "evaluate.html",
        {
            "request": request,
            "session_id": session_id,
        }
    )


@app.get("/summary/{session_id}", response_class=HTMLResponse)
async def summary_page(request: Request, session_id: str):
    """Summary page showing results after evaluation."""
    return templates.TemplateResponse(
        "summary.html",
        {
            "request": request,
            "session_id": session_id,
        }
    )


@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request):
    """History page showing past evaluations."""
    return templates.TemplateResponse(
        "history.html",
        {"request": request}
    )


@app.get("/analytics", response_class=HTMLResponse)
async def analytics_page(request: Request):
    """Analytics page showing aggregate statistics."""
    return templates.TemplateResponse(
        "analytics.html",
        {"request": request}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
