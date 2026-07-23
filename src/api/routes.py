"""API routes for XAUUSD Scalping System."""
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import pandas as pd
import numpy as np

from ..config import get_config
from ..data import DataLoader
from ..features import FeatureEngineer
from ..models.persistence import ModelPersistence
from ..signals.generator import SignalGenerator, SignalType

router = APIRouter()

# Global instances
_data_loader: Optional[DataLoader] = None
_feature_engineer: Optional[FeatureEngineer] = None
_signal_generator: Optional[SignalGenerator] = None
_model_persistence = ModelPersistence()


# Pydantic models
class SignalResponse(BaseModel):
    timestamp: datetime
    signal: str  # BUY, SELL, WAIT
    confidence: float
    price: float
    metadata: Dict[str, Any] = {}


class BacktestRequest(BaseModel):
    symbol: str = "XAUUSD=X"
    interval: str = "1m"
    period: str = "7d"
    model_name: str = "ensemble"
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class BacktestResponse(BaseModel):
    total_return: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    total_trades: int
    trades: List[Dict[str, Any]]


class TrainRequest(BaseModel):
    model_name: str = "ensemble"
    symbol: str = "XAUUSD=X"
    interval: str = "1m"
    period: str = "30d"
    test_size: float = 0.2
    validation_size: float = 0.1


class TrainResponse(BaseModel):
    model_name: str
    version: str
    metrics: Dict[str, float]
    feature_importance: List[Dict[str, Any]]


class SubsystemStatus(BaseModel):
    name: str
    status: str  # healthy, degraded, unhealthy, unknown
    message: str = ""
    details: Dict[str, Any] = {}


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    models_loaded: List[str]


class DiagnosticResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    uptime_seconds: float = 0
    environment: str = "production"
    subsystems: Dict[str, SubsystemStatus]
    environment_variables: Dict[str, str]
    issues: List[str]


# Dependency functions
def get_data_loader() -> DataLoader:
    global _data_loader
    if _data_loader is None:
        _data_loader = DataLoader()
    return _data_loader


def get_feature_engineer() -> FeatureEngineer:
    global _feature_engineer
    if _feature_engineer is None:
        _feature_engineer = FeatureEngineer()
    return _feature_engineer


def get_signal_generator() -> SignalGenerator:
    global _signal_generator
    if _signal_generator is None:
        # Load latest model
        persistence = ModelPersistence()
        try:
            loaded = persistence.load_latest("ensemble")
            _signal_generator = SignalGenerator(
                model=loaded['model'],
                scaler=loaded.get('scaler'),
                feature_names=loaded.get('feature_names'),
            )
        except Exception:
            _signal_generator = None
    return _signal_generator


# Routes
@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    persistence = ModelPersistence()
    models = persistence.list_models()
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0",
        models_loaded=models,
    )


@router.get("/diagnostic", response_model=DiagnosticResponse)
async def diagnostic_check():
    """
    Comprehensive diagnostic endpoint showing status of all subsystems.
    
    Returns detailed information about:
    - Data connection (Yahoo Finance)
    - Model loading
    - Feature generation
    - Signal generation
    - Telegram notification configuration
    - Environment variables
    """
    import os
    import time
    import logging
    from pathlib import Path
    
    start_time = time.time()
    subsystems = {}
    issues = []
    
    # Check 1: Configuration
    try:
        config = get_config()
        subsystems["config"] = SubsystemStatus(
            name="Configuration",
            status="healthy",
            message="Configuration loaded successfully",
            details={"config_path": config.config_path}
        )
    except Exception as e:
        subsystems["config"] = SubsystemStatus(
            name="Configuration",
            status="unhealthy",
            message=f"Failed to load config: {str(e)}"
        )
        issues.append(f"Configuration error: {str(e)}")
    
    # Check 2: Yahoo Finance Data Connection
    try:
        dl = DataLoader()
        test_df = dl.load_ohlcv(interval="1m", period="1d", use_cache=False)
        if test_df.empty:
            subsystems["data"] = SubsystemStatus(
                name="Yahoo Finance",
                status="degraded",
                message="Connected but returned empty data",
                details={"last_bar": None}
            )
            issues.append("Yahoo Finance returned empty data")
        else:
            subsystems["data"] = SubsystemStatus(
                name="Yahoo Finance",
                status="healthy",
                message=f"Connected, {len(test_df)} bars loaded",
                details={
                    "bars": len(test_df),
                    "last_bar": test_df.index[-1].isoformat() if not test_df.empty else None,
                    "symbol": "GC=F"
                }
            )
    except Exception as e:
        subsystems["data"] = SubsystemStatus(
            name="Yahoo Finance",
            status="unhealthy",
            message=f"Connection failed: {str(e)}",
            details={"error_type": type(e).__name__}
        )
        issues.append(f"Yahoo Finance connection error: {str(e)}")
    
    # Check 3: Models
    try:
        persistence = ModelPersistence()
        models = persistence.list_models()
        model_versions = {}
        for model_name in models:
            versions = persistence.list_versions(model_name)
            model_versions[model_name] = versions
        
        if not models:
            subsystems["models"] = SubsystemStatus(
                name="AI Models",
                status="degraded",
                message="No trained models found. Run training to generate signals.",
                details={"available_models": [], "model_dir": str(Path("models"))}
            )
            issues.append("No trained models - signals cannot be generated")
        else:
            subsystems["models"] = SubsystemStatus(
                name="AI Models",
                status="healthy",
                message=f"{len(models)} model type(s) available",
                details={"models": model_versions}
            )
    except Exception as e:
        subsystems["models"] = SubsystemStatus(
            name="AI Models",
            status="unhealthy",
            message=f"Model loading failed: {str(e)}",
            details={"error_type": type(e).__name__}
        )
        issues.append(f"Model loading error: {str(e)}")
    
    # Check 4: Feature Engineering
    try:
        fe = FeatureEngineer()
        feature_names = fe.get_feature_names()
        subsystems["features"] = SubsystemStatus(
            name="Feature Engineering",
            status="healthy",
            message=f"{len(feature_names)} features configured",
            details={"feature_count": len(feature_names)}
        )
    except Exception as e:
        subsystems["features"] = SubsystemStatus(
            name="Feature Engineering",
            status="unhealthy",
            message=f"Feature engine failed: {str(e)}",
            details={"error_type": type(e).__name__}
        )
        issues.append(f"Feature engineering error: {str(e)}")
    
    # Check 5: Telegram Configuration
    telegram_status = "unhealthy"
    telegram_message = "Not configured"
    telegram_details = {}
    
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    enabled = os.getenv("TELEGRAM_ENABLED", "false").lower() == "true"
    
    if not bot_token and not chat_id:
        telegram_message = "Telegram not configured (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID not set)"
        telegram_details = {"configured": False, "enabled": False}
    elif not bot_token:
        telegram_message = "TELEGRAM_BOT_TOKEN not set"
        telegram_details = {"configured": False, "bot_token_set": False}
    elif not chat_id:
        telegram_message = "TELEGRAM_CHAT_ID not set"
        telegram_details = {"configured": False, "chat_id_set": False}
    else:
        telegram_status = "healthy" if enabled else "degraded"
        telegram_message = "Telegram configured" if enabled else "Telegram configured but disabled"
        telegram_details = {"configured": True, "enabled": enabled}
    
    subsystems["telegram"] = SubsystemStatus(
        name="Telegram Notifications",
        status=telegram_status,
        message=telegram_message,
        details=telegram_details
    )
    
    if telegram_status == "unhealthy":
        issues.append("Telegram not configured - notifications disabled")
    
    # Check 6: Environment Variables
    env_vars = {}
    sensitive_keys = ["TELEGRAM_BOT_TOKEN", "API_KEY", "DISCORD_WEBHOOK_URL", "NEWS_API_KEY"]
    required_vars = ["PYTHONPATH", "CONFIG_PATH", "PORT"]
    optional_vars = [
        "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "TELEGRAM_ENABLED",
        "API_HOST", "API_PORT", "API_KEY",
        "DATA_SYMBOL", "DATA_INTERVAL",
        "LOG_LEVEL", "LOG_JSON_FORMAT"
    ]
    
    for var in required_vars + [v for v in optional_vars if v not in required_vars]:
        value = os.getenv(var, "")
        if var in sensitive_keys and value:
            env_vars[var] = "***" + value[-4:] if len(value) > 4 else "***"
        else:
            env_vars[var] = value if value else "(not set)"
    
    subsystems["environment"] = SubsystemStatus(
        name="Environment",
        status="healthy",
        message="Environment variables accessible",
        details={"var_count": len(env_vars)}
    )
    
    # Check 7: File System (directories)
    dirs_to_check = ["models", "data", "logs", "config"]
    dir_status = {}
    for dir_name in dirs_to_check:
        dir_path = Path("/app") / dir_name
        exists = dir_path.exists()
        writable = os.access(dir_path, os.W_OK) if exists else False
        dir_status[dir_name] = {"exists": exists, "writable": writable}
    
    all_dirs_ok = all(d["exists"] for d in dir_status.values())
    subsystems["filesystem"] = SubsystemStatus(
        name="File System",
        status="healthy" if all_dirs_ok else "degraded",
        message="All directories accessible" if all_dirs_ok else "Some directories missing",
        details=dir_status
    )
    
    # Determine overall status
    overall_status = "healthy"
    if any(s.status == "unhealthy" for s in subsystems.values()):
        overall_status = "degraded"
    if subsystems.get("data", SubsystemStatus(name="", status="healthy")).status == "unhealthy":
        overall_status = "unhealthy"
    
    return DiagnosticResponse(
        status=overall_status,
        timestamp=datetime.now(),
        version="1.0.0",
        uptime_seconds=time.time() - start_time,
        environment=os.getenv("ENVIRONMENT", "production"),
        subsystems=subsystems,
        environment_variables=env_vars,
        issues=issues
    )


@router.get("/status")
async def status_check():
    """Simple status check for load balancers."""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/signals/latest", response_model=SignalResponse)
async def get_latest_signal(
    data_loader: DataLoader = Depends(get_data_loader),
    feature_engineer: FeatureEngineer = Depends(get_feature_engineer),
    signal_generator: SignalGenerator = Depends(get_signal_generator),
):
    """Get latest trading signal."""
    if signal_generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Get latest data
    df = data_loader.load_ohlcv(interval="1m", period="1d")
    if df.empty:
        raise HTTPException(status_code=503, detail="No data available")
    
    # Generate features
    features = feature_engineer.transform(df.tail(100))
    if features.empty:
        raise HTTPException(status_code=503, detail="Feature generation failed")
    
    # Generate signal
    signals = signal_generator.generate(features.tail(1), df.tail(1))
    if not signals:
        raise HTTPException(status_code=503, detail="Signal generation failed")
    
    signal = signals[0]
    return SignalResponse(
        timestamp=signal.timestamp,
        signal=signal.signal_type.name,
        confidence=signal.confidence,
        price=signal.price,
        metadata=signal.metadata,
    )


@router.get("/signals/history", response_model=List[SignalResponse])
async def get_signal_history(
    hours: int = 24,
    data_loader: DataLoader = Depends(get_data_loader),
    feature_engineer: FeatureEngineer = Depends(get_feature_engineer),
    signal_generator: SignalGenerator = Depends(get_signal_generator),
):
    """Get historical signals."""
    if signal_generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Get data
    df = data_loader.load_ohlcv(interval="1m", period=f"{hours//60+1}d")
    if df.empty:
        raise HTTPException(status_code=503, detail="No data available")
    
    # Generate features
    features = feature_engineer.transform(df)
    if features.empty:
        raise HTTPException(status_code=503, detail="Feature generation failed")
    
    # Generate signals
    signals = signal_generator.generate(features, df)
    
    # Filter by time
    cutoff = pd.Timestamp.now() - pd.Timedelta(hours=hours)
    recent_signals = [s for s in signals if s.timestamp >= cutoff]
    
    return [
        SignalResponse(
            timestamp=s.timestamp,
            signal=s.signal_type.name,
            confidence=s.confidence,
            price=s.price,
            metadata=s.metadata,
        )
        for s in recent_signals
    ]


@router.post("/backtest", response_model=BacktestResponse)
async def run_backtest(
    request: BacktestRequest,
    background_tasks: BackgroundTasks,
):
    """Run backtest."""
    from backtesting.engine import BacktestEngine
    from models.trainer import ModelTrainer
    
    # Load data
    data_loader = DataLoader()
    if request.start_date and request.end_date:
        df = data_loader.load_ohlcv(
            interval=request.interval,
            start=request.start_date,
            end=request.end_date,
        )
    else:
        df = data_loader.load_ohlcv(interval=request.interval, period=request.period)
    
    if df.empty:
        raise HTTPException(status_code=400, detail="No data available for backtest")
    
    # Load model
    persistence = ModelPersistence()
    try:
        loaded = persistence.load_latest(request.model_name)
        model = loaded['model']
        scaler = loaded.get('scaler')
        feature_names = loaded.get('feature_names')
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Model not found: {e}")
    
    # Generate features
    feature_engineer = FeatureEngineer()
    features = feature_engineer.transform(df)
    
    if feature_names:
        features = features[feature_names]
    
    # Generate signals
    signal_generator = SignalGenerator(model=model, scaler=scaler, feature_names=feature_names)
    signals = signal_generator.generate(features, df)
    signal_series = signal_generator.get_signal_series(signals)
    confidence_series = signal_generator.get_confidence_series(signals)
    
    # Run backtest
    engine = BacktestEngine()
    results = engine.run(df, signal_series, confidence_series)
    
    # Prepare trade list
    trades = []
    if 'trades' in results and not results['trades'].empty:
        trades_df = results['trades']
        trades = trades_df.to_dict('records')
    
    return BacktestResponse(
        total_return=results.get('total_return', 0),
        win_rate=results.get('win_rate', 0),
        profit_factor=results.get('profit_factor', 0),
        max_drawdown=results.get('max_drawdown', 0),
        sharpe_ratio=results.get('sharpe_ratio', 0),
        total_trades=results.get('n_trades', 0),
        trades=trades,
    )


@router.post("/train", response_model=TrainResponse)
async def train_model(
    request: TrainRequest,
    background_tasks: BackgroundTasks,
):
    """Train new model."""
    from ..models.trainer import ModelTrainer
    from ..features import generate_training_data
    
    # Load data
    data_loader = DataLoader()
    df = data_loader.load_ohlcv(interval=request.interval, period=request.period)
    
    if df.empty:
        raise HTTPException(status_code=400, detail="No data available for training")
    
    # Generate features and targets
    feature_engineer = FeatureEngineer()
    features = feature_engineer.fit_transform(df)
    
    # Generate targets
    target_config = get_config().get_section('target')
    features, target = generate_training_data(
        df,
        target_horizon=target_config.get('horizon', 5),
        threshold_long=target_config.get('threshold_long', 0.0015),
        threshold_short=target_config.get('threshold_short', -0.0015),
        method=target_config.get('labeling_method', 'triple_barrier'),
    )
    
    # Train model
    trainer = ModelTrainer()
    results = trainer.train_all(features, target, test_size=request.test_size)
    
    # Get best model metrics
    best_model = trainer.best_model_name
    if best_model and best_model in results:
        best_metrics = results[best_model]
    else:
        best_metrics = {}
    
    # Save model
    persistence = ModelPersistence()
    version = persistence.save(
        model=trainer.models[best_model],
        model_name=request.model_name,
        scaler=trainer.scalers.get(best_model),
        feature_names=trainer.feature_names,
        metrics=best_metrics,
        feature_importance=trainer.get_feature_importance(best_model),
        config=trainer.config.get_section('models'),
    )
    
    # Get feature importance
    importance_df = trainer.get_feature_importance(best_model)
    feature_importance = importance_df.to_dict('records') if not importance_df.empty else []
    
    return TrainResponse(
        model_name=request.model_name,
        version=version.split('/')[-1],
        metrics=best_metrics,
        feature_importance=feature_importance,
    )


@router.get("/models")
async def list_models():
    """List available models."""
    persistence = ModelPersistence()
    models = {}
    for model_name in persistence.list_models():
        versions = persistence.list_versions(model_name)
        models[model_name] = versions
    return models


@router.get("/models/{model_name}/versions")
async def list_model_versions(model_name: str):
    """List versions for a model."""
    persistence = ModelPersistence()
    versions = persistence.list_versions(model_name)
    return versions


@router.get("/data/latest")
async def get_latest_data(
    interval: str = "1m",
    bars: int = 100,
    data_loader: DataLoader = Depends(get_data_loader),
):
    """Get latest market data."""
    df = data_loader.get_latest_data(interval=interval, n_bars=bars)
    if df.empty:
        raise HTTPException(status_code=503, detail="No data available")
    
    return df.to_dict('records')


@router.get("/data/ohlcv")
async def get_ohlcv_data(
    interval: str = "1m",
    period: str = "1d",
    data_loader: DataLoader = Depends(get_data_loader),
):
    """Get OHLCV data."""
    df = data_loader.load_ohlcv(interval=interval, period=period)
    if df.empty:
        raise HTTPException(status_code=503, detail="No data available")
    
    return df.to_dict('records')


@router.get("/features/names")
async def get_feature_names(
    feature_engineer: FeatureEngineer = Depends(get_feature_engineer),
):
    """Get feature names."""
    names = feature_engineer.get_feature_names()
    return {"features": names, "count": len(names)}


@router.post("/signals/generate")
async def generate_signals(
    interval: str = "1m",
    period: str = "1d",
    data_loader: DataLoader = Depends(get_data_loader),
    feature_engineer: FeatureEngineer = Depends(get_feature_engineer),
    signal_generator: SignalGenerator = Depends(get_signal_generator),
):
    """Generate signals for recent data."""
    if signal_generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    df = data_loader.load_ohlcv(interval=interval, period=period)
    if df.empty:
        raise HTTPException(status_code=503, detail="No data available")
    
    features = feature_engineer.transform(df)
    if features.empty:
        raise HTTPException(status_code=503, detail="Feature generation failed")
    
    signals = signal_generator.generate(features, df)
    signal_series = signal_generator.get_signal_series(signals)
    confidence_series = signal_generator.get_confidence_series(signals)
    
    # Combine with OHLCV
    result = pd.DataFrame({
        'signal': signal_series,
        'confidence': confidence_series,
    }).join(df[['open', 'high', 'low', 'close', 'volume']])
    
    return result.to_dict('records')


if __name__ == "__main__":
    # Test routes
    print("API routes module loaded")
