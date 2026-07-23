"""FastAPI application for XAUUSD Scalping System."""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from ..config import get_config
from .routes import router


def get_port() -> int:
    """Get port from environment variable."""
    port = os.getenv("PORT")
    if port:
        return int(port)
    return 8000


def get_host() -> str:
    """Get host from environment variable."""
    return os.getenv("API_HOST", "0.0.0.0")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan."""
    # Startup
    config = get_config()
    api_config = config.get_section('api')
    
    host = get_host()
    port = get_port()
    print(f"Starting XAUUSD Scalping API on {host}:{port}")
    
    yield
    
    # Shutdown
    print("Shutting down XAUUSD Scalping API")


app = FastAPI(
    title="XAUUSD AI Scalping System API",
    description="REST API for AI-powered XAUUSD scalping signals",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
config = get_config()
api_config = config.get_section('api')
app.add_middleware(
    CORSMiddleware,
    allow_origins=api_config.get('cors_origins', ['*']),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "XAUUSD AI Scalping System API",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=get_host(),
        port=get_port(),
        workers=1,  # Use 1 worker in single process mode
        reload=False,
    )
