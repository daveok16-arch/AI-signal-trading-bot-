"""Volume features for XAUUSD Scalping System."""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class VolumeFeatures:
    """Volume-based feature calculations."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.indicators = self.config.get('indicators', [
            'obv', 'vwap', 'mfi', 'cmf', 'eom', 'nvi', 'pvi', 'vpt', 'vwma'
        ])
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate volume features."""
        features = pd.DataFrame(index=df.index)
        
        open_, high, low, close, volume = df['open'], df['high'], df['low'], df['close'], df['volume']
        
        if 'obv' in self.indicators:
            features['obv'] = self._obv(close, volume)
            features['obv_ma'] = features['obv'].rolling(20).mean()
            features['obv_slope'] = features['obv'].diff(5)
        
        if 'vwap' in self.indicators:
            features['vwap'] = self._vwap(high, low, close, volume)
            features['vwap_dist'] = (close - features['vwap']) / features['vwap']
            features['vwap_slope'] = features['vwap'].diff(5)
        
        if 'mfi' in self.indicators:
            features['mfi_14'] = self._mfi(high, low, close, volume, 14)
            features['mfi_21'] = self._mfi(high, low, close, volume, 21)
        
        if 'cmf' in self.indicators:
            features['cmf_20'] = self._cmf(high, low, close, volume, 20)
        
        if 'eom' in self.indicators:
            features['eom_14'] = self._eom(high, low, close, volume, 14)
        
        if 'nvi' in self.indicators:
            features['nvi'] = self._nvi(close, volume)
        
        if 'pvi' in self.indicators:
            features['pvi'] = self._pvi(close, volume)
        
        if 'vpt' in self.indicators:
            features['vpt'] = self._vpt(close, volume)
        
        if 'vwma' in self.indicators:
            features['vwma_20'] = self._vwma(close, volume, 20)
            features['vwma_50'] = self._vwma(close, volume, 50)
            features['vwma_ratio'] = features['vwma_20'] / (features['vwma_50'] + 1e-10)
        
        # Additional volume features
        features['volume'] = volume
        features['volume_ma_5'] = volume.rolling(5).mean()
        features['volume_ma_20'] = volume.rolling(20).mean()
        features['volume_ma_50'] = volume.rolling(50).mean()
        features['volume_ratio_5_20'] = volume / (features['volume_ma_20'] + 1e-10)
        features['volume_ratio_20_50'] = features['volume_ma_20'] / (features['volume_ma_50'] + 1e-10)
        
        # Volume rate of change
        features['volume_roc_1'] = volume.pct_change(1)
        features['volume_roc_5'] = volume.pct_change(5)
        features['volume_roc_20'] = volume.pct_change(20)
        
        # Volume volatility
        features['volume_std_20'] = volume.rolling(20).std()
        features['volume_cv_20'] = features['volume_std_20'] / (features['volume_ma_20'] + 1e-10)
        
        # Buy/Sell volume (approximate)
        features['buy_volume'] = volume * (close > open_).astype(int)
        features['sell_volume'] = volume * (close < open_).astype(int)
        features['buy_sell_ratio'] = features['buy_volume'].rolling(20).sum() / (features['sell_volume'].rolling(20).sum() + 1e-10)
        
        # Volume-weighted price
        features['vwap_session'] = self._session_vwap(high, low, close, volume)
        
        return features
    
    def _obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume."""
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv
    
    def _vwap(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Volume Weighted Average Price."""
        tp = (high + low + close) / 3
        return (tp * volume).cumsum() / volume.cumsum()
    
    def _mfi(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int) -> pd.Series:
        """Money Flow Index."""
        tp = (high + low + close) / 3
        money_flow = tp * volume
        
        positive_flow = money_flow.where(tp > tp.shift(1), 0).rolling(period).sum()
        negative_flow = money_flow.where(tp < tp.shift(1), 0).rolling(period).sum()
        
        mfi = 100 - (100 / (1 + positive_flow / (negative_flow + 1e-10)))
        return mfi
    
    def _cmf(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int) -> pd.Series:
        """Chaikin Money Flow."""
        mfv = ((close - low) - (high - close)) / (high - low + 1e-10) * volume
        cmf = mfv.rolling(period).sum() / volume.rolling(period).sum()
        return cmf
    
    def _eom(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int) -> pd.Series:
        """Ease of Movement."""
        distance = (high + low) / 2 - (high.shift(1) + low.shift(1)) / 2
        box_ratio = volume / (high - low + 1e-10)
        eom = distance / (box_ratio + 1e-10)
        return eom.rolling(period).mean()
    
    def _nvi(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Negative Volume Index."""
        nvi = pd.Series(1000.0, index=close.index)
        for i in range(1, len(close)):
            if volume.iloc[i] < volume.iloc[i-1]:
                nvi.iloc[i] = nvi.iloc[i-1] * (close.iloc[i] / close.iloc[i-1])
            else:
                nvi.iloc[i] = nvi.iloc[i-1]
        return nvi
    
    def _pvi(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Positive Volume Index."""
        pvi = pd.Series(1000.0, index=close.index)
        for i in range(1, len(close)):
            if volume.iloc[i] > volume.iloc[i-1]:
                pvi.iloc[i] = pvi.iloc[i-1] * (close.iloc[i] / close.iloc[i-1])
            else:
                pvi.iloc[i] = pvi.iloc[i-1]
        return pvi
    
    def _vpt(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Volume Price Trend."""
        vpt = (volume * close.pct_change()).cumsum()
        return vpt
    
    def _vwma(self, close: pd.Series, volume: pd.Series, period: int) -> pd.Series:
        """Volume Weighted Moving Average."""
        vwma = (close * volume).rolling(period).sum() / volume.rolling(period).sum()
        return vwma
    
    def _session_vwap(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Session VWAP (resets at midnight UTC)."""
        tp = (high + low + close) / 3
        # Detect session breaks (new day)
        new_session = (high.index.hour == 0) & (high.index.minute == 0)
        # Handle both Series and array cases
        new_session_vals = new_session.values if hasattr(new_session, 'values') else new_session
        tp_vals = tp.values if hasattr(tp, 'values') else tp
        volume_vals = volume.values if hasattr(volume, 'values') else volume
        close_vals = close.values if hasattr(close, 'values') else close
        
        vwap = np.zeros(len(close))
        cum_tpv = 0.0
        cum_vol = 0.0
        
        for i in range(len(close)):
            if new_session_vals[i] or i == 0:
                cum_tpv = tp_vals[i] * volume_vals[i]
                cum_vol = volume_vals[i]
            else:
                cum_tpv += tp_vals[i] * volume_vals[i]
                cum_vol += volume_vals[i]
            
            vwap[i] = cum_tpv / cum_vol if cum_vol > 0 else close_vals[i]
        
        return pd.Series(vwap, index=close.index)


if __name__ == "__main__":
    import yfinance as yf
    df = yf.download('GC=F', period='5d', interval='1h', progress=False)
    df.columns = [c.lower() for c in df.columns]
    
    vf = VolumeFeatures()
    features = vf.transform(df)
    print(f"Volume features: {len(features.columns)}")
    print(features.tail())
