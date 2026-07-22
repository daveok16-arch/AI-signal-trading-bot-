"""Integration tests for the full pipeline."""
import pandas as pd
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.loader import DataLoader
from src.features import FeatureEngineer
from src.models.trainer import ModelTrainer
from src.signals.generator import SignalGenerator
from src.backtesting.engine import BacktestEngine


def create_sample_data(n=5000):
    """Create realistic sample OHLCV data."""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=n, freq='1min')
    
    # Generate price with trend and noise
    trend = np.linspace(0, 10, n)
    noise = np.cumsum(np.random.randn(n) * 0.1)
    close = 2000 + trend + noise
    
    # Generate OHLC
    spread = np.abs(np.random.randn(n) * 0.05)
    high = close + spread
    low = close - spread
    open_ = close + np.random.randn(n) * 0.02
    volume = np.random.randint(100, 1000, n)
    
    df = pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    }, index=dates)
    
    return df


class TestFullPipeline:
    """Test complete pipeline."""
    
    @pytest.fixture
    def sample_data(self):
        return create_sample_data(5000)
    
    def test_data_loader(self, sample_data):
        """Test data loader."""
        # We can't easily test Yahoo Finance without network
        # But we can test the DataLoader class initialization
        loader = DataLoader()
        assert loader is not None
        assert loader.default_interval == '1m'
    
    def test_feature_engineering(self, sample_data):
        """Test feature engineering pipeline."""
        engineer = FeatureEngineer()
        features = engineer.transform(sample_data)
        
        assert len(features) == len(sample_data)
        assert len(features.columns) > 50  # Should have many features
        assert not features.isna().all().all()  # Not all NaN
    
    def test_model_training(self, sample_data):
        """Test model training."""
        engineer = FeatureEngineer()
        features = engineer.transform(sample_data)
        
        # Create dummy target
        target = pd.Series(
            np.random.choice([-1, 0, 1], len(features), p=[0.2, 0.6, 0.2]),
            index=features.index
        )
        
        # Align
        common = features.index.intersection(target.index)
        features = features.loc[common]
        target = target.loc[common]
        
        trainer = ModelTrainer()
        results = trainer.train_all(features, target, test_size=0.2, validation_size=0.1)
        
        assert len(results) > 0
        assert trainer.best_model_name is not None
    
    def test_signal_generation(self, sample_data):
        """Test signal generation."""
        engineer = FeatureEngineer()
        features = engineer.transform(sample_data)
        
        # Create dummy model
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000)
        
        # Create target and fit
        target = pd.Series(
            np.random.choice([0, 1, 2], len(features)),
            index=features.index
        )
        model.fit(features, target)
        
        # Generate signals
        generator = SignalGenerator(model=model)
        signals = generator.generate(features, sample_data)
        
        assert len(signals) == len(features)
        signal_series = generator.get_signal_series(signals)
        assert len(signal_series) == len(signals)
    
    def test_backtest_engine(self, sample_data):
        """Test backtest engine."""
        # Create simple signals
        np.random.seed(42)
        signals = pd.Series(
            np.random.choice([-1, 0, 1], len(sample_data), p=[0.1, 0.8, 0.1]),
            index=sample_data.index
        )
        confidence = pd.Series(
            np.random.uniform(0.5, 1.0, len(sample_data)),
            index=sample_data.index
        )
        
        engine = BacktestEngine()
        results = engine.run(sample_data, signals, confidence)
        
        assert 'total_return' in results
        assert 'n_trades' in results
        assert 'win_rate' in results
        assert 'max_drawdown' in results
        assert results['n_trades'] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
