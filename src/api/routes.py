"""API routes for XAUUSD Scalping System."""
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import pandas as pd
import numpy as np

from src.config import get_config
from src.data import DataLoader
from src.features import FeatureEngineer
from src.models.persistence import ModelPersistence
from src.signals.generator import SignalGenerator, SignalType

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


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    models_loaded: List[str]


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
    from src.backtesting.engine import BacktestEngine
    from src.models.trainer import ModelTrainer
    
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
    from src.models.trainer import ModelTrainer
    from features import generate_training_data
    
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
