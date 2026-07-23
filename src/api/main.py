"""FastAPI application for XAUUSD Scalping System."""
import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from ..config import get_config
from .routes import router

logger = logging.getLogger(__name__)

# Required environment variables for production
REQUIRED_ENV_VARS = [
    ("PYTHONPATH", str),
    ("CONFIG_PATH", str),
]

# Optional but recommended environment variables
RECOMMENDED_ENV_VARS = [
    ("TELEGRAM_BOT_TOKEN", str),
    ("TELEGRAM_CHAT_ID", str),
    ("TELEGRAM_ENABLED", bool),
    ("API_HOST", str),
    ("API_PORT", int),
    ("LOG_LEVEL", str),
    ("LOG_JSON_FORMAT", bool),
    ("DATA_SYMBOL", str),
    ("DATA_INTERVAL", str),
]


def get_port() -> int:
    """Get port from environment variable."""
    port = os.getenv("PORT")
    if port:
        return int(port)
    return 8000


def get_host() -> str:
    """Get host from environment variable."""
    return os.getenv("API_HOST", "0.0.0.0")


def validate_environment() -> Dict[str, Any]:
    """
    Validate environment variables and return validation report.
    
    Returns:
        Dictionary with:
        - is_valid: bool
        - errors: List[str] of critical errors
        - warnings: List[str] of warnings
        - info: Dict of environment variable values
    """
    errors = []
    warnings = []
    info = {}
    
    # Check required variables
    for var_name, var_type in REQUIRED_ENV_VARS:
        value = os.getenv(var_name)
        if value is None:
            # Check for common Render path patterns
            if var_name == "PYTHONPATH":
                # On Render, PYTHONPATH might be set differently
                if os.getenv("HOME") and "/opt/render" in os.getenv("HOME", ""):
                    warnings.append(f"{var_name} not set, but detected Render environment")
                else:
                    errors.append(f"Required environment variable {var_name} is not set")
            else:
                errors.append(f"Required environment variable {var_name} is not set")
        else:
            info[var_name] = value
    
    # Check recommended variables
    for var_name, var_type in RECOMMENDED_ENV_VARS:
        value = os.getenv(var_name)
        if value is not None:
            if var_type == bool:
                info[var_name] = value.lower() in ('true', '1', 'yes', 'on')
            else:
                info[var_name] = value
        else:
            info[var_name] = "(not set)"
    
    # Check config file exists
    config_path = os.getenv("CONFIG_PATH", "config/config.yaml")
    if not Path(config_path).exists():
        # Try relative to app directory
        alt_path = Path("/app") / config_path if config_path.startswith("config/") else Path("/app/config/config.yaml")
        if not alt_path.exists():
            errors.append(f"Config file not found: {config_path}")
        else:
            info["config_path_resolved"] = str(alt_path)
    
    # Check Telegram configuration
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if bot_token and not chat_id:
        warnings.append("TELEGRAM_BOT_TOKEN set but TELEGRAM_CHAT_ID not set")
    if chat_id and not bot_token:
        warnings.append("TELEGRAM_CHAT_ID set but TELEGRAM_BOT_TOKEN not set")
    if bot_token and chat_id:
        info["telegram_configured"] = True
    else:
        info["telegram_configured"] = False
        warnings.append("Telegram not configured - notifications will be disabled")
    
    # Validate port
    port = os.getenv("PORT", os.getenv("API_PORT"))
    if port:
        try:
            port_int = int(port)
            if port_int < 1 or port_int > 65535:
                errors.append(f"Invalid PORT value: {port} (must be 1-65535)")
        except ValueError:
            errors.append(f"Invalid PORT value: {port} (must be integer)")
    
    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "info": info,
    }


def print_environment_report(validation: Dict[str, Any]):
    """Print environment validation report."""
    print("\n" + "=" * 60)
    print("ENVIRONMENT VALIDATION REPORT")
    print("=" * 60)
    
    if validation["is_valid"]:
        print("✅ All required environment variables are set")
    else:
        print("❌ Environment validation failed:")
        for error in validation["errors"]:
            print(f"   ERROR: {error}")
    
    if validation["warnings"]:
        print("\n⚠️  Warnings:")
        for warning in validation["warnings"]:
            print(f"   - {warning}")
    
    print("\n📋 Environment Info:")
    for key, value in validation["info"].items():
        if "token" in key.lower() and value != "(not set)" and value is not True:
            value = "***" + value[-4:] if len(str(value)) > 4 else "***"
        print(f"   {key}: {value}")
    
    print("=" * 60 + "\n")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan."""
    # Validate environment at startup
    validation = validate_environment()
    print_environment_report(validation)
    
    # Log warnings
    for warning in validation["warnings"]:
        logger.warning(f"Startup warning: {warning}")
    
    # Fail fast if critical errors
    if not validation["is_valid"]:
        for error in validation["errors"]:
            logger.error(f"Startup error: {error}")
        # Don't fail the API entirely, but log prominently
        logger.critical("API started with configuration errors - some features may not work")
    
    # Store validation in app state
    app.state.env_validation = validation
    
    # Startup
    config = get_config()
    api_config = config.get_section('api')
    
    host = get_host()
    port = get_port()
    logger.info(f"Starting XAUUSD Scalping API on {host}:{port}")
    print(f"Starting XAUUSD Scalping API on {host}:{port}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down XAUUSD Scalping API")
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
