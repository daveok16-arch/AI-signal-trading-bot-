"""Unit tests for feature engineering."""
import pytest
import pandas as pd
import numpy as np
from src.features.technical import TechnicalIndicators
from src.features.price_action import PriceActionFeatures
from src.features.market_structure import MarketStructureFeatures
from src.features.volatility import VolatilityFeatures
from src.features.momentum import MomentumFeatures
from src.features.volume import VolumeFeatures
from src.features.selection import FeatureSelector, EnsembleFeatureSelector


class TestTechnicalIndicators:
    """Test technical indicators."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        np.random.seed(42)
        n = 200
        close = 2000 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)
        open_ = close + np.random.randn(n) * 0.2
        volume = np.random.randint(100, 1000, n)
        
        return pd.DataFrame({
            'open': open_,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=pd.date_range('2024-01-01', periods=n, freq='1min'))
    
    def test_sma(self, sample_data):
        """Test SMA calculation."""
        sma = TechnicalIndicators.sma(sample_data['close'], 20)
        assert len(sma) == len(sample_data)
        assert not sma.isna().all()
    
    def test_ema(self, sample_data):
        """Test EMA calculation."""
        ema = TechnicalIndicators.ema(sample_data['close'], 20)
        assert len(ema) == len(sample_data)
        assert not ema.isna().all()
    
    def test_rsi(self, sample_data):
        """Test RSI calculation."""
        rsi = TechnicalIndicators.rsi(sample_data['close'], 14)
        assert len(rsi) == len(sample_data)
        # RSI should be in range [0, 100] for non-NaN values
        rsi_valid = rsi.dropna()
        assert (rsi_valid >= 0).all() and (rsi_valid <= 100).all()
    
    def test_macd(self, sample_data):
        """Test MACD calculation."""
        macd = TechnicalIndicators.macd(sample_data['close'])
        assert 'macd' in macd
        assert 'signal' in macd
        assert 'histogram' in macd
    
    def test_bollinger_bands(self, sample_data):
        """Test Bollinger Bands."""
        bb = TechnicalIndicators.bollinger_bands(sample_data['close'])
        assert 'upper' in bb
        assert 'middle' in bb
        assert 'lower' in bb
        assert 'bandwidth' in bb
        assert 'percent_b' in bb
    
    def test_atr(self, sample_data):
        """Test ATR calculation."""
        atr = TechnicalIndicators.atr(
            sample_data['high'], sample_data['low'], sample_data['close']
        )
        assert len(atr) == len(sample_data)
        # ATR should be non-negative for non-NaN values
        atr_valid = atr.dropna()
        assert (atr_valid >= 0).all()
    
    def test_all_indicators(self, sample_data):
        """Test all indicators together."""
        result = TechnicalIndicators.all_indicators(sample_data)
        assert len(result.columns) > len(sample_data.columns)


class TestPriceActionFeatures:
    """Test price action features."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        np.random.seed(42)
        n = 200
        close = 2000 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)
        open_ = close + np.random.randn(n) * 0.2
        volume = np.random.randint(100, 1000, n)
        
        return pd.DataFrame({
            'open': open_,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=pd.date_range('2024-01-01', periods=n, freq='1min'))
    
    def test_transform(self, sample_data):
        """Test price action transform."""
        pa = PriceActionFeatures()
        features = pa.transform(sample_data)
        
        assert len(features) == len(sample_data)
        assert 'body_size' in features.columns
        assert 'upper_wick' in features.columns
        assert 'lower_wick' in features.columns
        assert 'is_bullish' in features.columns
        assert 'is_bearish' in features.columns


class TestMarketStructureFeatures:
    """Test market structure features."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        np.random.seed(42)
        n = 200
        close = 2000 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)
        
        return pd.DataFrame({
            'high': high,
            'low': low,
            'close': close
        }, index=pd.date_range('2024-01-01', periods=n, freq='1min'))
    
    def test_transform(self, sample_data):
        """Test market structure transform."""
        ms = MarketStructureFeatures()
        features = ms.transform(sample_data)
        
        assert len(features) == len(sample_data)
        assert 'swing_high' in features.columns
        assert 'swing_low' in features.columns
        assert 'bos_bullish' in features.columns
        assert 'bos_bearish' in features.columns


class TestVolatilityFeatures:
    """Test volatility features."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        np.random.seed(42)
        n = 200
        close = 2000 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)
        
        return pd.DataFrame({
            'open': close,
            'high': high,
            'low': low,
            'close': close
        }, index=pd.date_range('2024-01-01', periods=n, freq='1min'))
    
    def test_transform(self, sample_data):
        """Test volatility transform."""
        vf = VolatilityFeatures()
        features = vf.transform(sample_data)
        
        assert len(features) == len(sample_data)
        assert 'atr_14' in features.columns
        assert 'natr_14' in features.columns


class TestMomentumFeatures:
    """Test momentum features."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        np.random.seed(42)
        n = 200
        close = 2000 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)
        volume = np.random.randint(100, 1000, n)
        
        return pd.DataFrame({
            'open': close,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=pd.date_range('2024-01-01', periods=n, freq='1min'))
    
    def test_transform(self, sample_data):
        """Test momentum transform."""
        mf = MomentumFeatures()
        features = mf.transform(sample_data)
        
        assert len(features) == len(sample_data)
        assert 'roc_5' in features.columns
        assert 'mom_10' in features.columns
        assert 'tsi' in features.columns


class TestVolumeFeatures:
    """Test volume features."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        np.random.seed(42)
        n = 200
        close = 2000 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)
        volume = np.random.randint(100, 1000, n)
        
        return pd.DataFrame({
            'open': close,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=pd.date_range('2024-01-01', periods=n, freq='1min'))
    
    def test_transform(self, sample_data):
        """Test volume transform."""
        vf = VolumeFeatures()
        features = vf.transform(sample_data)
        
        assert len(features) == len(sample_data)
        assert 'obv' in features.columns
        assert 'vwap' in features.columns
        assert 'mfi_14' in features.columns


class TestFeatureSelection:
    """Test feature selection."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample features and target."""
        np.random.seed(42)
        n = 500
        X = pd.DataFrame(
            np.random.randn(n, 50),
            columns=[f'feat_{i}' for i in range(50)],
            index=pd.date_range('2024-01-01', periods=n, freq='1min')
        )
        # Add some informative features
        X['feat_0'] = X['feat_0'] + np.random.choice([-1, 0, 1], n, p=[0.3, 0.4, 0.3])
        y = pd.Series(np.random.choice([0, 1, 2], n), index=X.index)
        return X, y
    
    def test_feature_selector(self, sample_data):
        """Test FeatureSelector."""
        X, y = sample_data
        selector = FeatureSelector({'method': 'mutual_info', 'k_best': 20})
        X_selected = selector.fit_transform(X, y)
        
        assert len(X_selected.columns) <= 20
        assert len(selector.selected_features) > 0
    
    def test_ensemble_selector(self, sample_data):
        """Test EnsembleFeatureSelector."""
        X, y = sample_data
        selector = EnsembleFeatureSelector({'k_best': 20, 'vote_threshold': 2})
        X_selected = selector.fit_transform(X, y)
        
        assert len(X_selected.columns) <= 20
        assert len(selector.selected_features) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
