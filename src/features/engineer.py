"""Main feature engineering pipeline for XAUUSD Scalping System."""
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Any, Tuple
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from src.config import get_config
from data import DataLoader
from .price_action import PriceActionFeatures
from .market_structure import MarketStructureFeatures
from .volatility import VolatilityFeatures
from .momentum import MomentumFeatures
from .volume import VolumeFeatures
from .selection import FeatureSelector
from src.data.cache import get_feature_cache

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Main feature engineering pipeline."""
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        feature_version: str = "v1",
    ):
        self.config = get_config(config_path)
        self.feature_version = feature_version
        self.feature_cache = get_feature_cache()
        
        # Initialize feature modules
        features_config = self.config.get_section('features')
        
        self.price_action = PriceActionFeatures(
            config=features_config.get('price_action', {}),
        ) if features_config.get('price_action', {}).get('enabled', True) else None
        
        self.market_structure = MarketStructureFeatures(
            config=features_config.get('market_structure', {}),
        ) if features_config.get('market_structure', {}).get('enabled', True) else None
        
        self.volatility = VolatilityFeatures(
            config=features_config.get('volatility', {}),
        ) if features_config.get('volatility', {}).get('enabled', True) else None
        
        self.momentum = MomentumFeatures(
            config=features_config.get('momentum', {}),
        ) if features_config.get('momentum', {}).get('enabled', True) else None
        
        self.volume = VolumeFeatures(
            config=features_config.get('volume', {}),
        ) if features_config.get('volume', {}).get('enabled', True) else None
        
        self.selector = FeatureSelector(
            config=features_config.get('feature_selection', {}),
        ) if features_config.get('feature_selection', {}).get('enabled', True) else None
        
        # Technical indicators
        self.technical_config = features_config.get('technical', {})
    
    def transform(
        self,
        df: pd.DataFrame,
        fit_selector: bool = False,
        target: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Transform OHLCV data into feature matrix.
        
        Args:
            df: OHLCV DataFrame with columns [open, high, low, close, volume]
            fit_selector: Whether to fit feature selector (training only)
            target: Target series for feature selection (training only)
            
        Returns:
            Feature DataFrame
        """
        if df.empty:
            return pd.DataFrame()
        
        logger.info(f"Generating features for {len(df)} bars")
        
        # Make copy
        df = df.copy()
        
        # Ensure required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Generate all feature groups
        feature_dfs = []
        
        # 1. Technical indicators
        if self.technical_config.get('enabled', True):
            tech_features = self._generate_technical_features(df)
            feature_dfs.append(tech_features)
            logger.debug(f"Technical features: {len(tech_features.columns)}")
        
        # 2. Price action patterns
        if self.price_action:
            pa_features = self.price_action.transform(df)
            feature_dfs.append(pa_features)
            logger.debug(f"Price action features: {len(pa_features.columns)}")
        
        # 3. Market structure
        if self.market_structure:
            ms_features = self.market_structure.transform(df)
            feature_dfs.append(ms_features)
            logger.debug(f"Market structure features: {len(ms_features.columns)}")
        
        # 4. Volatility features
        if self.volatility:
            vol_features = self.volatility.transform(df)
            feature_dfs.append(vol_features)
            logger.debug(f"Volatility features: {len(vol_features.columns)}")
        
        # 5. Momentum features
        if self.momentum:
            mom_features = self.momentum.transform(df)
            feature_dfs.append(mom_features)
            logger.debug(f"Momentum features: {len(mom_features.columns)}")
        
        # 6. Volume features
        if self.volume:
            vol_features = self.volume.transform(df)
            feature_dfs.append(vol_features)
            logger.debug(f"Volume features: {len(vol_features.columns)}")
        
        # Combine all features
        if feature_dfs:
            features = pd.concat(feature_dfs, axis=1)
        else:
            features = pd.DataFrame(index=df.index)
        
        # Clean features
        features = self._clean_features(features)
        
        # Feature selection (if enabled)
        if self.selector and target is not None:
            features = self.selector.transform(features, target, fit=fit_selector)
            logger.info(f"After feature selection: {len(features.columns)} features")
        
        logger.info(f"Generated {len(features.columns)} features")
        return features
    
    def _generate_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate technical indicator features."""
        from src.features.technical import TechnicalIndicators
        
        tech = TechnicalIndicators()
        indicators = tech.all_indicators(df, config=self.technical_config)
        
        # Return only new features (not original OHLCV)
        original_cols = {'open', 'high', 'low', 'close', 'volume', 'returns', 'log_returns', 
                        'hl_spread', 'oc_spread', 'volume_ma', 'volume_ratio', 'close_position'}
        new_cols = [c for c in indicators.columns if c not in original_cols]
        
        return indicators[new_cols]
    
    def _clean_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Clean feature matrix."""
        features = features.copy()
        
        # Remove constant columns
        nunique = features.nunique()
        constant_cols = nunique[nunique <= 1].index.tolist()
        if constant_cols:
            logger.debug(f"Removing {len(constant_cols)} constant columns")
            features = features.drop(columns=constant_cols)
        
        # Remove highly correlated features
        corr_threshold = self.config.get('features.feature_selection.correlation_threshold', 0.95)
        if len(features.columns) > 1:
            corr_matrix = features.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [col for col in upper.columns if any(upper[col] > corr_threshold)]
            if to_drop:
                logger.debug(f"Removing {len(to_drop)} highly correlated columns")
                features = features.drop(columns=to_drop)
        
        # Handle infinities
        features = features.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill then backfill NaN
        features = features.ffill().bfill()
        
        # Fill any remaining NaN with 0
        features = features.fillna(0)
        
        return features
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        target: pd.Series,
    ) -> pd.DataFrame:
        """Fit and transform (for training)."""
        return self.transform(df, fit_selector=True, target=target)
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        # Create dummy data to get feature names
        dummy_df = pd.DataFrame({
            'open': np.random.randn(100).cumsum() + 2000,
            'high': np.random.randn(100).cumsum() + 2005,
            'low': np.random.randn(100).cumsum() + 1995,
            'close': np.random.randn(100).cumsum() + 2000,
            'volume': np.random.randint(100, 1000, 100),
        }, index=pd.date_range('2024-01-01', periods=100, freq='1min'))
        
        dummy_df['high'] = dummy_df[['open', 'high', 'close']].max(axis=1)
        dummy_df['low'] = dummy_df[['open', 'low', 'close']].min(axis=1)
        
        features = self.transform(dummy_df)
        return features.columns.tolist()
    
    def save_features(
        self,
        features: pd.DataFrame,
        symbol: str = "XAUUSD",
        timeframe: str = "1m",
    ) -> bool:
        """Save features to cache."""
        return self.feature_cache.save_features(
            symbol, timeframe, features, self.feature_version
        )
    
    def load_features(
        self,
        symbol: str = "XAUUSD",
        timeframe: str = "1m",
    ) -> Optional[pd.DataFrame]:
        """Load features from cache."""
        return self.feature_cache.load_features(symbol, timeframe, self.feature_version)


def generate_training_data(
    df: pd.DataFrame,
    target_horizon: int = 5,
    threshold_long: float = 0.0015,
    threshold_short: float = -0.0015,
    method: str = 'triple_barrier',
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generate features and targets for training.
    
    Args:
        df: OHLCV DataFrame
        target_horizon: Minutes ahead to predict
        threshold_long: Long threshold
        threshold_short: Short threshold
        method: Labeling method ('simple', 'triple_barrier')
        
    Returns:
        (features, target) tuple
    """
    engineer = FeatureEngineer()
    features = engineer.fit_transform(df, target=None)
    
    # Generate targets
    if method == 'triple_barrier':
        target = create_triple_barrier_labels(
            df, target_horizon, threshold_long, threshold_short
        )
    else:
        target = create_simple_labels(df, target_horizon, threshold_long, threshold_short)
    
    # Align
    common_idx = features.index.intersection(target.index)
    features = features.loc[common_idx]
    target = target.loc[common_idx]
    
    return features, target


def create_simple_labels(
    df: pd.DataFrame,
    horizon: int,
    threshold_long: float,
    threshold_short: float,
) -> pd.Series:
    """Create simple forward return labels."""
    future_return = df['close'].pct_change(horizon).shift(-horizon)
    
    labels = pd.Series(0, index=df.index, dtype=int)  # 0 = wait, 1 = long, -1 = short
    labels[future_return > threshold_long] = 1
    labels[future_return < threshold_short] = -1
    
    return labels


def create_triple_barrier_labels(
    df: pd.DataFrame,
    horizon: int,
    threshold_long: float,
    threshold_short: float,
    vol_lookback: int = 20,
    upper_mult: float = 2.0,
    lower_mult: float = 2.0,
) -> pd.Series:
    """Create triple barrier labels (advocate for labeling)."""
    from mlfinlab.labeling import get_events, get_bins
    
    close = df['close']
    volatility = close.pct_change().rolling(vol_lookback).std()
    
    # Daily volatility
    daily_vol = volatility * np.sqrt(1440)  # 1440 minutes in a day
    
    # Events (every bar)
    events = pd.Series(close.index, index=close.index)
    
    # Barriers
    upper = daily_vol * upper_mult
    lower = daily_vol * lower_mult
    
    # Get labels
    try:
        labels = get_bins(events, close, upper, lower, daily_vol)
        return labels['bin']
    except Exception as e:
        logger.warning(f"Triple barrier labeling failed: {e}, falling back to simple")
        return create_simple_labels(df, horizon, threshold_long, threshold_short)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test feature engineering
    import yfinance as yf
    df = yf.download('GC=F', period='5d', interval='1h', progress=False)
    df.columns = [c.lower() for c in df.columns]
    
    engineer = FeatureEngineer()
    features = engineer.transform(df)
    print(f"Generated {len(features.columns)} features")
    print(features.columns.tolist()[:20])
    print(features.tail())
