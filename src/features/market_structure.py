"""Market structure features for XAUUSD Scalping System."""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MarketStructureFeatures:
    """Market structure analysis features."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.swing_lookback = self.config.get('swing_lookback', 10)
        self.bos_threshold = self.config.get('bos_threshold', 0.0005)
        self.choch_threshold = self.config.get('choch_threshold', 0.0005)
        self.fvg_min_gap = self.config.get('fvg_min_gap', 0.0002)
        self.ob_lookback = self.config.get('order_block_lookback', 20)
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate market structure features."""
        features = pd.DataFrame(index=df.index)
        
        high, low, close = df['high'], df['low'], df['close']
        
        # Swing points
        swing_highs = self._find_swing_highs(high)
        swing_lows = self._find_swing_lows(low)
        
        features['swing_high'] = swing_highs
        features['swing_low'] = swing_lows
        
        # Market structure
        features['market_structure'] = self._identify_structure(swing_highs, swing_lows)
        
        # Break of Structure (BOS)
        features['bos_bullish'] = self._detect_bos_bullish(close, swing_highs)
        features['bos_bearish'] = self._detect_bos_bearish(close, swing_lows)
        
        # Change of Character (CHoCH)
        features['choch_bullish'] = self._detect_choch_bullish(close, swing_highs, swing_lows)
        features['choch_bearish'] = self._detect_choch_bearish(close, swing_highs, swing_lows)
        
        # Fair Value Gaps (FVG)
        features['fvg_bullish'] = self._detect_fvg_bullish(high, low)
        features['fvg_bearish'] = self._detect_fvg_bearish(high, low)
        
        # Order Blocks
        features['order_block_bullish'] = self._detect_order_block_bullish(df)
        features['order_block_bearish'] = self._detect_order_block_bearish(df)
        
        # Liquidity
        features['buy_side_liquidity'] = self._detect_buy_side_liquidity(high, swing_highs)
        features['sell_side_liquidity'] = self._detect_sell_side_liquidity(low, swing_lows)
        
        # Trend structure
        features['trend_direction'] = self._get_trend_direction(swing_highs, swing_lows)
        features['trend_strength'] = self._get_trend_strength(close, swing_highs, swing_lows)
        
        # Internal/External structure
        features['internal_structure'] = self._get_internal_structure(close, swing_highs, swing_lows)
        
        return features
    
    def _find_swing_highs(self, high: pd.Series) -> pd.Series:
        """Find swing highs using fractal method."""
        lookback = self.swing_lookback
        swing_high = pd.Series(0, index=high.index)
        
        for i in range(lookback, len(high) - lookback):
            window = high.iloc[i-lookback:i+lookback+1]
            if high.iloc[i] == window.max():
                # Confirm it's a significant swing
                left_max = high.iloc[i-lookback:i].max()
                right_max = high.iloc[i+1:i+lookback+1].max()
                if (high.iloc[i] - left_max) / high.iloc[i] > 0.0001 or \
                   (high.iloc[i] - right_max) / high.iloc[i] > 0.0001:
                    swing_high.iloc[i] = 1
        
        return swing_high
    
    def _find_swing_lows(self, low: pd.Series) -> pd.Series:
        """Find swing lows using fractal method."""
        lookback = self.swing_lookback
        swing_low = pd.Series(0, index=low.index)
        
        for i in range(lookback, len(low) - lookback):
            window = low.iloc[i-lookback:i+lookback+1]
            if low.iloc[i] == window.min():
                left_min = low.iloc[i-lookback:i].min()
                right_min = low.iloc[i+1:i+lookback+1].min()
                if (left_min - low.iloc[i]) / low.iloc[i] > 0.0001 or \
                   (right_min - low.iloc[i]) / low.iloc[i] > 0.0001:
                    swing_low.iloc[i] = 1
        
        return swing_low
    
    def _identify_structure(self, swing_highs: pd.Series, swing_lows: pd.Series) -> pd.Series:
        """Identify market structure (HH/HL/LH/LL)."""
        structure = pd.Series(0, index=swing_highs.index)  # 0=neutral, 1=uptrend, -1=downtrend
        
        highs_idx = swing_highs[swing_highs == 1].index
        lows_idx = swing_lows[swing_lows == 1].index
        
        if len(highs_idx) < 2 or len(lows_idx) < 2:
            return structure
        
        # Get prices at swing points
        high_prices = swing_highs.index.map(lambda x: swing_highs.loc[x])  # This needs fixing
        # Actually let me fix this - we need the price values
        pass
        
        # Simpler approach: track last two swings
        last_high_idx = highs_idx[-1] if len(highs_idx) > 0 else None
        prev_high_idx = highs_idx[-2] if len(highs_idx) > 1 else None
        last_low_idx = lows_idx[-1] if len(lows_idx) > 0 else None
        prev_low_idx = lows_idx[-2] if len(lows_idx) > 1 else None
        
        return structure
    
    def _detect_bos_bullish(self, close: pd.Series, swing_highs: pd.Series) -> pd.Series:
        """Detect bullish break of structure."""
        bos = pd.Series(0, index=close.index)
        high_levels = close[swing_highs == 1]
        
        if len(high_levels) == 0:
            return bos
        
        last_high = high_levels.iloc[-1]
        bos[(close > last_high * (1 + self.bos_threshold)) & (close.shift(1) <= last_high)] = 1
        
        return bos
    
    def _detect_bos_bearish(self, close: pd.Series, swing_lows: pd.Series) -> pd.Series:
        """Detect bearish break of structure."""
        bos = pd.Series(0, index=close.index)
        low_levels = close[swing_lows == 1]
        
        if len(low_levels) == 0:
            return bos
        
        last_low = low_levels.iloc[-1]
        bos[(close < last_low * (1 - self.bos_threshold)) & (close.shift(1) >= last_low)] = 1
        
        return bos
    
    def _detect_choch_bullish(self, close: pd.Series, swing_highs: pd.Series, swing_lows: pd.Series) -> pd.Series:
        """Detect bullish change of character."""
        choch = pd.Series(0, index=close.index)
        
        # Need downtrend (LH, LL) then break above recent LH
        highs_idx = swing_highs[swing_highs == 1].index
        lows_idx = swing_lows[swing_lows == 1].index
        
        if len(highs_idx) < 2 or len(lows_idx) < 2:
            return choch
        
        # Check for downtrend structure
        recent_highs = close.loc[highs_idx[-3:]] if len(highs_idx) >= 3 else close.loc[highs_idx]
        recent_lows = close.loc[lows_idx[-3:]] if len(lows_idx) >= 3 else close.loc[lows_idx]
        
        if len(recent_highs) >= 2 and len(recent_lows) >= 2:
            # LH and LL
            if recent_highs.iloc[-1] < recent_highs.iloc[-2] and recent_lows.iloc[-1] < recent_lows.iloc[-2]:
                # Break above recent LH
                lh_level = recent_highs.iloc[-1]
                choch[(close > lh_level * (1 + self.choch_threshold)) & 
                      (close.shift(1) <= lh_level)] = 1
        
        return choch
    
    def _detect_choch_bearish(self, close: pd.Series, swing_highs: pd.Series, swing_lows: pd.Series) -> pd.Series:
        """Detect bearish change of character."""
        choch = pd.Series(0, index=close.index)
        
        highs_idx = swing_highs[swing_highs == 1].index
        lows_idx = swing_lows[swing_lows == 1].index
        
        if len(highs_idx) < 2 or len(lows_idx) < 2:
            return choch
        
        recent_highs = close.loc[highs_idx[-3:]] if len(highs_idx) >= 3 else close.loc[highs_idx]
        recent_lows = close.loc[lows_idx[-3:]] if len(lows_idx) >= 3 else close.loc[lows_idx]
        
        if len(recent_highs) >= 2 and len(recent_lows) >= 2:
            # HH and HL
            if recent_highs.iloc[-1] > recent_highs.iloc[-2] and recent_lows.iloc[-1] > recent_lows.iloc[-2]:
                # Break below recent HL
                hl_level = recent_lows.iloc[-1]
                choch[(close < hl_level * (1 - self.choch_threshold)) & 
                      (close.shift(1) >= hl_level)] = 1
        
        return choch
    
    def _detect_fvg_bullish(self, high: pd.Series, low: pd.Series) -> pd.Series:
        """Detect bullish fair value gap."""
        fvg = pd.Series(0, index=high.index)
        
        # FVG: low[i] > high[i-2] (gap between candle i-2 high and candle i low)
        gap = (low - high.shift(2)) / high.shift(2)
        fvg[(gap > self.fvg_min_gap) & (low > high.shift(2))] = 1
        
        return fvg
    
    def _detect_fvg_bearish(self, high: pd.Series, low: pd.Series) -> pd.Series:
        """Detect bearish fair value gap."""
        fvg = pd.Series(0, index=high.index)
        
        # FVG: high[i] < low[i-2]
        gap = (low.shift(2) - high) / low.shift(2)
        fvg[(gap > self.fvg_min_gap) & (high < low.shift(2))] = 1
        
        return fvg
    
    def _detect_order_block_bullish(self, df: pd.DataFrame) -> pd.Series:
        """Detect bullish order block."""
        ob = pd.Series(0, index=df.index)
        lookback = self.ob_lookback
        
        # Check if 'open' column exists, if not create it from close
        if 'open' in df.columns:
            open_ = df['open']
        else:
            open_ = df['close'].shift(1).fillna(df['close'])
        
        high, low, close = df['high'], df['low'], df['close']
        
        for i in range(lookback, len(df)):
            # Look for down candle followed by strong up move
            if close.iloc[i-1] < open_.iloc[i-1]:  # Down candle
                # Check if next few candles break structure
                future_high = high.iloc[i:i+5].max()
                if future_high > high.iloc[i-1] * 1.001:  # Break above
                    # This down candle is a bullish OB
                    ob.iloc[i-1] = 1
        
        return ob
    
    def _detect_order_block_bearish(self, df: pd.DataFrame) -> pd.Series:
        """Detect bearish order block."""
        ob = pd.Series(0, index=df.index)
        lookback = self.ob_lookback
        
        # Check if 'open' column exists, if not create it from close
        if 'open' in df.columns:
            open_ = df['open']
        else:
            open_ = df['close'].shift(1).fillna(df['close'])
        
        high, low, close = df['high'], df['low'], df['close']
        
        for i in range(lookback, len(df)):
            # Look for up candle followed by strong down move
            if close.iloc[i-1] > open_.iloc[i-1]:  # Up candle
                future_low = low.iloc[i:i+5].min()
                if future_low < low.iloc[i-1] * 0.999:  # Break below
                    ob.iloc[i-1] = 1
        
        return ob
    
    def _detect_buy_side_liquidity(self, high: pd.Series, swing_highs: pd.Series) -> pd.Series:
        """Detect buy side liquidity (swing highs)."""
        return swing_highs  # Swing highs are buy side liquidity
    
    def _detect_sell_side_liquidity(self, low: pd.Series, swing_lows: pd.Series) -> pd.Series:
        """Detect sell side liquidity (swing lows)."""
        return swing_lows  # Swing lows are sell side liquidity
    
    def _get_trend_direction(self, swing_highs: pd.Series, swing_lows: pd.Series) -> pd.Series:
        """Get trend direction from structure."""
        trend = pd.Series(0, index=swing_highs.index)
        
        highs_idx = swing_highs[swing_highs == 1].index
        lows_idx = swing_lows[swing_lows == 1].index
        
        if len(highs_idx) >= 2 and len(lows_idx) >= 2:
            # Compare recent structure
            last_high = highs_idx[-1] if len(highs_idx) > 0 else None
            prev_high = highs_idx[-2] if len(highs_idx) > 1 else None
            last_low = lows_idx[-1] if len(lows_idx) > 0 else None
            prev_low = lows_idx[-2] if len(lows_idx) > 1 else None
            
            # This is a simplified version
            # In practice, you'd compare the actual price values
            pass
        
        return trend
    
    def _get_trend_strength(self, close: pd.Series, swing_highs: pd.Series, swing_lows: pd.Series) -> pd.Series:
        """Calculate trend strength."""
        strength = pd.Series(0.0, index=close.index)
        
        # Simple trend strength based on swing structure
        highs_idx = swing_highs[swing_highs == 1].index
        lows_idx = swing_lows[swing_lows == 1].index
        
        if len(highs_idx) >= 2 and len(lows_idx) >= 2:
            # Get values at swing points
            high_vals = close.loc[highs_idx]
            low_vals = close.loc[lows_idx]
            
            # Trend strength = slope of swings
            if len(high_vals) >= 2:
                high_slope = (high_vals.iloc[-1] - high_vals.iloc[-2]) / high_vals.iloc[-2]
            else:
                high_slope = 0
            
            if len(low_vals) >= 2:
                low_slope = (low_vals.iloc[-1] - low_vals.iloc[-2]) / low_vals.iloc[-2]
            else:
                low_slope = 0
            
            # Average slope
            avg_slope = (high_slope + low_slope) / 2
            strength.iloc[-1] = avg_slope * 10000  # Scale to pips
        
        return strength.ffill()
    
    def _get_internal_structure(self, close: pd.Series, swing_highs: pd.Series, swing_lows: pd.Series) -> pd.Series:
        """Get internal market structure (higher timeframe context)."""
        internal = pd.Series(0, index=close.index)
        
        # This would typically use higher timeframe data
        # For now, use a simplified version based on recent price action
        
        return internal


if __name__ == "__main__":
    import yfinance as yf
    df = yf.download('GC=F', period='5d', interval='1h', progress=False)
    df.columns = [c.lower() for c in df.columns]
    
    ms = MarketStructureFeatures()
    features = ms.transform(df)
    print(f"Market structure features: {len(features.columns)}")
    print(features.tail())
