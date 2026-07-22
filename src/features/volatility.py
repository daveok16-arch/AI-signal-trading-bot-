"""Volatility features for XAUUSD Scalping System."""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class VolatilityFeatures:
    """Volatility feature calculations."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.indicators = self.config.get('indicators', [
            'atr', 'natr', 'keltner', 'donchian', 
            'historical_vol', 'parkinson_vol', 'garman_klass_vol', 'yang_zhang_vol'
        ])
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate volatility features."""
        features = pd.DataFrame(index=df.index)
        
        high, low, close = df['high'], df['low'], df['close']
        
        if 'atr' in self.indicators:
            features['atr_14'] = self._atr(high, low, close, 14)
            features['atr_21'] = self._atr(high, low, close, 21)
            features['atr_ratio_14_21'] = features['atr_14'] / (features['atr_21'] + 1e-10)
        
        if 'natr' in self.indicators:
            features['natr_14'] = self._natr(high, low, close, 14)
        
        if 'keltner' in self.indicators:
            keltner = self._keltner_channels(high, low, close)
            features['keltner_upper'] = keltner['upper']
            features['keltner_middle'] = keltner['middle']
            features['keltner_lower'] = keltner['lower']
            features['keltner_width'] = (keltner['upper'] - keltner['lower']) / keltner['middle']
            features['keltner_pos'] = (close - keltner['lower']) / (keltner['upper'] - keltner['lower'] + 1e-10)
        
        if 'donchian' in self.indicators:
            donchian = self._donchian_channels(high, low)
            features['donchian_upper'] = donchian['upper']
            features['donchian_lower'] = donchian['lower']
            features['donchian_middle'] = donchian['middle']
            features['donchian_width'] = (donchian['upper'] - donchian['lower']) / donchian['middle']
            features['donchian_pos'] = (close - donchian['lower']) / (donchian['upper'] - donchian['lower'] + 1e-10)
        
        if 'historical_vol' in self.indicators:
            for period in [10, 20, 50]:
                features[f'hist_vol_{period}'] = self._historical_volatility(close, period)
        
        if 'parkinson_vol' in self.indicators:
            features['parkinson_vol_10'] = self._parkinson_volatility(high, low, 10)
            features['parkinson_vol_20'] = self._parkinson_volatility(high, low, 20)
        
        if 'garman_klass_vol' in self.indicators:
            features['gk_vol_10'] = self._garman_klass_volatility(high, low, close, 10)
            features['gk_vol_20'] = self._garman_klass_volatility(high, low, close, 20)
        
        if 'yang_zhang_vol' in self.indicators:
            features['yz_vol_10'] = self._yang_zhang_volatility(df['open'], high, low, close, 10)
            features['yz_vol_20'] = self._yang_zhang_volatility(df['open'], high, low, close, 20)
        
        # Additional volatility features
        features['realized_vol_5'] = self._realized_volatility(close, 5)
        features['realized_vol_15'] = self._realized_volatility(close, 15)
        features['realized_vol_30'] = self._realized_volatility(close, 30)
        
        # Volatility regime
        features['vol_regime'] = self._volatility_regime(features)
        
        # Volatility ratios
        if 'hist_vol_20' in features.columns and 'hist_vol_50' in features.columns:
            features['vol_ratio_20_50'] = features['hist_vol_20'] / (features['hist_vol_50'] + 1e-10)
        
        # GARCH-like features (simple)
        returns = close.pct_change()
        features['abs_returns'] = returns.abs()
        features['sq_returns'] = returns ** 2
        features['vol_ma_5'] = features['abs_returns'].rolling(5).mean()
        features['vol_ma_20'] = features['abs_returns'].rolling(20).mean()
        features['vol_of_vol'] = features['abs_returns'].rolling(20).std()
        
        return features
    
    def _atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """Average True Range."""
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()
    
    def _natr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """Normalized ATR."""
        atr = self._atr(high, low, close, period)
        return (atr / close) * 100
    
    def _keltner_channels(
        self, 
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series,
        period: int = 20,
        atr_period: int = 10,
        multiplier: float = 2.0,
    ) -> Dict[str, pd.Series]:
        """Keltner Channels."""
        ema = close.ewm(span=period, adjust=False).mean()
        atr = self._atr(high, low, close, atr_period)
        
        upper = ema + multiplier * atr
        lower = ema - multiplier * atr
        
        return {'upper': upper, 'middle': ema, 'lower': lower}
    
    def _donchian_channels(
        self,
        high: pd.Series,
        low: pd.Series,
        period: int = 20,
    ) -> Dict[str, pd.Series]:
        """Donchian Channels."""
        upper = high.rolling(window=period).max()
        lower = low.rolling(window=period).min()
        middle = (upper + lower) / 2
        
        return {'upper': upper, 'middle': middle, 'lower': lower}
    
    def _historical_volatility(self, close: pd.Series, period: int) -> pd.Series:
        """Historical volatility (standard deviation of returns)."""
        returns = np.log(close / close.shift(1))
        return returns.rolling(window=period).std() * np.sqrt(1440)  # Annualized for 1min
    
    def _parkinson_volatility(self, high: pd.Series, low: pd.Series, period: int) -> pd.Series:
        """Parkinson volatility estimator."""
        hl_ratio = np.log(high / low)
        parkinson = np.sqrt(hl_ratio ** 2 / (4 * np.log(2)))
        return parkinson.rolling(window=period).mean() * np.sqrt(1440)
    
    def _garman_klass_volatility(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int,
    ) -> pd.Series:
        """Garman-Klass volatility estimator."""
        log_hl = np.log(high / low)
        log_co = np.log(close / close.shift(1))
        
        gk = np.sqrt(
            0.5 * log_hl ** 2 - 
            (2 * np.log(2) - 1) * log_co ** 2
        )
        
        return gk.rolling(window=period).mean() * np.sqrt(1440)
    
    def _yang_zhang_volatility(
        self,
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int,
    ) -> pd.Series:
        """Yang-Zhang volatility estimator."""
        log_oc = np.log(open_ / close.shift(1))
        log_ho = np.log(high / open_)
        log_lo = np.log(low / open_)
        log_co = np.log(close / open_)
        
        # Overnight volatility
        overnight = log_oc ** 2
        
        # Open-to-close volatility
        open_close = log_co ** 2
        
        # Rogers-Satchell volatility
        rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
        
        k = 0.34 / (1.34 + (period + 1) / (period - 1))
        
        yz = overnight + k * open_close + (1 - k) * rs
        yz = np.sqrt(yz.rolling(window=period).mean()) * np.sqrt(1440)
        
        return yz
    
    def _realized_volatility(self, close: pd.Series, period: int) -> pd.Series:
        """Realized volatility (sum of squared returns)."""
        returns = close.pct_change()
        rv = (returns ** 2).rolling(window=period).sum()
        return np.sqrt(rv * 1440 / period)
    
    def _volatility_regime(self, features: pd.DataFrame) -> pd.Series:
        """Determine volatility regime."""
        regime = pd.Series(0, index=features.index)
        
        if 'hist_vol_20' in features.columns:
            vol = features['hist_vol_20']
            vol_ma = vol.rolling(50).mean()
            vol_std = vol.rolling(50).std()
            
            regime[(vol > vol_ma + vol_std)] = 2  # High vol
            regime[(vol < vol_ma - vol_std)] = -2  # Low vol
            regime[(vol > vol_ma) & (vol <= vol_ma + vol_std)] = 1  # Medium-high
            regime[(vol < vol_ma) & (vol >= vol_ma - vol_std)] = -1  # Medium-low
        
        return regime


if __name__ == "__main__":
    import yfinance as yf
    df = yf.download('GC=F', period='5d', interval='1h', progress=False)
    df.columns = [c.lower() for c in df.columns]
    
    vf = VolatilityFeatures()
    features = vf.transform(df)
    print(f"Volatility features: {len(features.columns)}")
    print(features.tail())
