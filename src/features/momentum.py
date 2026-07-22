"""Momentum features for XAUUSD Scalping System."""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class MomentumFeatures:
    """Momentum indicator features."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.indicators = self.config.get('indicators', [
            'roc', 'mom', 'tsi', 'uo', 'kst', 'stoch_rsi'
        ])
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate momentum features."""
        features = pd.DataFrame(index=df.index)
        
        close = df['close']
        
        if 'roc' in self.indicators:
            for period in [1, 5, 10, 20, 50]:
                features[f'roc_{period}'] = self._roc(close, period)
        
        if 'mom' in self.indicators:
            for period in [10, 20]:
                features[f'mom_{period}'] = self._momentum(close, period)
        
        if 'tsi' in self.indicators:
            features['tsi'] = self._tsi(close)
        
        if 'uo' in self.indicators:
            features['uo'] = self._ultimate_oscillator(df['high'], df['low'], close)
        
        if 'kst' in self.indicators:
            features['kst'] = self._kst(close)
        
        if 'stoch_rsi' in self.indicators:
            stoch_rsi = self._stoch_rsi(close)
            features['stoch_rsi_k'] = stoch_rsi['k']
            features['stoch_rsi_d'] = stoch_rsi['d']
        
        # Additional momentum features
        features['price_velocity'] = close.diff(1)
        features['price_acceleration'] = features['price_velocity'].diff(1)
        
        # Rate of change of ROC
        features['roc_roc_10'] = features.get('roc_10', self._roc(close, 10)).diff(1)
        
        # Momentum divergence
        features['mom_divergence'] = self._momentum_divergence(close)
        
        # Relative momentum vs MA
        for ma_period in [20, 50]:
            ma = close.rolling(ma_period).mean()
            features[f'mom_vs_ma{ma_period}'] = (close - ma) / ma
        
        return features
    
    def _roc(self, close: pd.Series, period: int) -> pd.Series:
        """Rate of Change."""
        return ((close - close.shift(period)) / close.shift(period)) * 100
    
    def _momentum(self, close: pd.Series, period: int) -> pd.Series:
        """Momentum (price difference)."""
        return close - close.shift(period)
    
    def _tsi(self, close: pd.Series, long: int = 25, short: int = 13) -> pd.Series:
        """True Strength Index."""
        momentum = close.diff(1)
        
        abs_momentum = momentum.abs()
        
        # Double smoothing
        first_smooth = momentum.ewm(span=long, adjust=False).mean()
        second_smooth = first_smooth.ewm(span=short, adjust=False).mean()
        
        abs_first = abs_momentum.ewm(span=long, adjust=False).mean()
        abs_second = abs_first.ewm(span=short, adjust=False).mean()
        
        tsi = 100 * (second_smooth / (abs_second + 1e-10))
        return tsi
    
    def _ultimate_oscillator(
        self, 
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series,
        period1: int = 7, 
        period2: int = 14, 
        period3: int = 28,
    ) -> pd.Series:
        """Ultimate Oscillator."""
        bp = close - pd.concat([low, close.shift(1)], axis=1).min(axis=1)
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        
        avg1 = bp.rolling(period1).sum() / tr.rolling(period1).sum()
        avg2 = bp.rolling(period2).sum() / tr.rolling(period2).sum()
        avg3 = bp.rolling(period3).sum() / tr.rolling(period3).sum()
        
        uo = 100 * (4 * avg1 + 2 * avg2 + avg3) / (4 + 2 + 1)
        return uo
    
    def _kst(self, close: pd.Series) -> pd.Series:
        """Know Sure Thing oscillator."""
        # Rate of change for 4 periods
        roc1 = self._roc(close, 10)
        roc2 = self._roc(close, 15)
        roc3 = self._roc(close, 20)
        roc4 = self._roc(close, 30)
        
        # SMA of ROCs
        sma1 = roc1.rolling(10).mean()
        sma2 = roc2.rolling(10).mean()
        sma3 = roc3.rolling(10).mean()
        sma4 = roc4.rolling(15).mean()
        
        # KST
        kst = (sma1 * 1 + sma2 * 2 + sma3 * 3 + sma4 * 4)
        return kst
    
    def _stoch_rsi(self, close: pd.Series, period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> Dict[str, pd.Series]:
        """Stochastic RSI."""
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        # Stochastic on RSI
        rsi_min = rsi.rolling(period).min()
        rsi_max = rsi.rolling(period).max()
        stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10)
        
        k = stoch_rsi.rolling(smooth_k).mean() * 100
        d = k.rolling(smooth_d).mean()
        
        return {'k': k, 'd': d}
    
    def _momentum_divergence(self, close: pd.Series, lookback: int = 20) -> pd.Series:
        """Detect momentum divergence."""
        # Price momentum
        price_mom = close.diff(lookback)
        
        # RSI momentum
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain / (loss + 1e-10)))
        rsi_mom = rsi.diff(lookback)
        
        # Divergence: price makes higher high but RSI makes lower high (bearish)
        # or price makes lower low but RSI makes higher low (bullish)
        divergence = pd.Series(0, index=close.index)
        
        # Simple divergence detection
        price_hh = (close > close.shift(1)) & (close.shift(1) > close.shift(2))
        rsi_lh = (rsi < rsi.shift(1)) & (rsi.shift(1) < rsi.shift(2))
        
        # Handle NaN values properly
        bearish_div = (price_hh & rsi_lh).fillna(False)
        # Use shift with fill_value to avoid NaN issues
        bearish_shifted = bearish_div.shift(-2, fill_value=False)
        divergence.loc[bearish_shifted] = -1  # Bearish divergence
        
        price_ll = (close < close.shift(1)) & (close.shift(1) < close.shift(2))
        rsi_hl = (rsi > rsi.shift(1)) & (rsi.shift(1) > rsi.shift(2))
        
        bullish_div = (price_ll & rsi_hl).fillna(False)
        bullish_shifted = bullish_div.shift(-2, fill_value=False)
        divergence.loc[bullish_shifted] = 1  # Bullish divergence
        
        return divergence


if __name__ == "__main__":
    import yfinance as yf
    df = yf.download('GC=F', period='5d', interval='1h', progress=False)
    df.columns = [c.lower() for c in df.columns]
    
    mf = MomentumFeatures()
    features = mf.transform(df)
    print(f"Momentum features: {len(features.columns)}")
    print(features.tail())
