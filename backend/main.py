from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.detection.service import get_detection_service
from app.routes.analysis import router as analysis_router
from app.routes.protection import router as protection_router


@asynccontextmanager
async def lifespan(_app: FastAPI):
    get_detection_service()
    yield

app = FastAPI(
    title="Social Media Privacy Guard API",
    description="Local AI detection API for context-aware image anonymization.",
    version="0.9.0",
    lifespan=lifespan,
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
    expose_headers=["X-Protection-Metadata", "Content-Disposition"],
)

app.include_router(analysis_router)
app.include_router(protection_router)


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
