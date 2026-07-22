"""FastAPI application for XAUUSD Scalping System."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from config import get_config
from .routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan."""
    # Startup
    config = get_config()
    api_config = config.get_section('api')
    
    print(f"Starting XAUUSD Scalping API on {api_config.get('host', '0.0.0.0')}:{api_config.get('port', 8000)}")
    
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
        host=api_config.get('host', '0.0.0.0'),
        port=api_config.get('port', 8000),
        workers=api_config.get('workers', 4),
        reload=False,
    )
