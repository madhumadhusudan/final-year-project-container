from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.analysis import router as analysis_router

app = FastAPI(
    title="Social Media Privacy Guard API",
    description="Local AI detection API for context-aware image anonymization.",
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(analysis_router)


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "message": "Social Media Privacy Guard API",
        "status": "running",
    }


@app.get("/health")
async def health() -> dict[str, str]:
    return {
        "status": "healthy",
        "service": "privacy-guard-backend",
    }
