"""Price action pattern features for XAUUSD Scalping System."""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class PriceActionFeatures:
    """Price action pattern detection and features."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.patterns = self.config.get('patterns', [
            'pinbar', 'engulfing', 'doji', 'hammer', 'shooting_star',
            'morning_star', 'evening_star', 'three_white_soldiers',
            'three_black_crows', 'inside_bar', 'outside_bar'
        ])
        self.swing_lookback = self.config.get('swing_detection', {}).get('lookback', 5)
        self.min_swing_pct = self.config.get('swing_detection', {}).get('min_swing_pct', 0.001)
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate price action features."""
        features = pd.DataFrame(index=df.index)
        
        open_, high, low, close = df['open'], df['high'], df['low'], df['close']
        
        # Basic candle properties
        features['body_size'] = (close - open_).abs() / close
        features['upper_wick'] = (high - close.where(close > open_, open_)) / close
        features['lower_wick'] = (close.where(close < open_, open_) - low) / close
        features['total_range'] = (high - low) / close
        features['body_to_range'] = features['body_size'] / (features['total_range'] + 1e-10)
        features['is_bullish'] = (close > open_).astype(int)
        features['is_bearish'] = (close < open_).astype(int)
        features['is_doji'] = (features['body_size'] < features['total_range'] * 0.1).astype(int)
        
        # Candle patterns
        if 'pinbar' in self.patterns:
            features['pinbar_bullish'] = self._detect_pinbar_bullish(open_, high, low, close)
            features['pinbar_bearish'] = self._detect_pinbar_bearish(open_, high, low, close)
        
        if 'engulfing' in self.patterns:
            features['engulfing_bullish'] = self._detect_engulfing_bullish(open_, high, low, close)
            features['engulfing_bearish'] = self._detect_engulfing_bearish(open_, high, low, close)
        
        if 'hammer' in self.patterns:
            features['hammer'] = self._detect_hammer(open_, high, low, close)
            features['inverted_hammer'] = self._detect_inverted_hammer(open_, high, low, close)
        
        if 'shooting_star' in self.patterns:
            features['shooting_star'] = self._detect_shooting_star(open_, high, low, close)
        
        if 'doji' in self.patterns:
            features['doji'] = self._detect_doji(open_, high, low, close)
            features['doji_gravestone'] = self._detect_gravestone_doji(open_, high, low, close)
            features['doji_dragonfly'] = self._detect_dragonfly_doji(open_, high, low, close)
        
        if 'morning_star' in self.patterns:
            features['morning_star'] = self._detect_morning_star(open_, high, low, close)
            features['evening_star'] = self._detect_evening_star(open_, high, low, close)
        
        if 'three_white_soldiers' in self.patterns:
            features['three_white_soldiers'] = self._detect_three_white_soldiers(open_, high, low, close)
            features['three_black_crows'] = self._detect_three_black_crows(open_, high, low, close)
        
        if 'inside_bar' in self.patterns:
            features['inside_bar'] = self._detect_inside_bar(high, low)
        
        if 'outside_bar' in self.patterns:
            features['outside_bar'] = self._detect_outside_bar(high, low)
        
        # Swing highs/lows
        features['swing_high'] = self._detect_swing_high(high)
        features['swing_low'] = self._detect_swing_low(low)
        
        # Distance to swings
        features['dist_to_swing_high'] = self._distance_to_swing(high, features['swing_high'])
        features['dist_to_swing_low'] = self._distance_to_swing(low, features['swing_low'])
        
        # Trend structure
        features['higher_high'] = self._higher_high(high)
        features['higher_low'] = self._higher_low(low)
        features['lower_high'] = self._lower_high(high)
        features['lower_low'] = self._lower_low(low)
        
        # Support/Resistance levels
        features['near_resistance'] = self._near_level(close, features['swing_high'], threshold=0.002)
        features['near_support'] = self._near_level(close, features['swing_low'], threshold=0.002)
        
        # Gap detection
        features['gap_up'] = (open_ > close.shift(1) * 1.001).astype(int)
        features['gap_down'] = (open_ < close.shift(1) * 0.999).astype(int)
        
        return features
    
    def _detect_pinbar_bullish(self, open_, high, low, close) -> pd.Series:
        """Detect bullish pinbar."""
        body = (close - open_).abs()
        lower_wick = open_.where(close > open_, close) - low
        upper_wick = high - close.where(close > open_, open_)
        
        return ((lower_wick > 2 * body) & (upper_wick < 0.5 * body) & (close > open_)).astype(int)
    
    def _detect_pinbar_bearish(self, open_, high, low, close) -> pd.Series:
        """Detect bearish pinbar."""
        body = (close - open_).abs()
        lower_wick = open_.where(close > open_, close) - low
        upper_wick = high - close.where(close > open_, open_)
        
        return ((upper_wick > 2 * body) & (lower_wick < 0.5 * body) & (close < open_)).astype(int)
    
    def _detect_engulfing_bullish(self, open_, high, low, close) -> pd.Series:
        """Detect bullish engulfing."""
        prev_open, prev_close = open_.shift(1), close.shift(1)
        return ((prev_close < prev_open) & (close > open_) & 
                (open_ < prev_close) & (close > prev_open)).astype(int)
    
    def _detect_engulfing_bearish(self, open_, high, low, close) -> pd.Series:
        """Detect bearish engulfing."""
        prev_open, prev_close = open_.shift(1), close.shift(1)
        return ((prev_close > prev_open) & (close < open_) & 
                (open_ > prev_close) & (close < prev_open)).astype(int)
    
    def _detect_hammer(self, open_, high, low, close) -> pd.Series:
        """Detect hammer."""
        body = (close - open_).abs()
        lower_wick = open_.where(close > open_, close) - low
        upper_wick = high - close.where(close > open_, open_)
        
        return ((lower_wick > 2 * body) & (upper_wick < 0.3 * body) & 
                (close > open_) & (body > 0)).astype(int)
    
    def _detect_inverted_hammer(self, open_, high, low, close) -> pd.Series:
        """Detect inverted hammer."""
        body = (close - open_).abs()
        lower_wick = open_.where(close > open_, close) - low
        upper_wick = high - close.where(close > open_, open_)
        
        return ((upper_wick > 2 * body) & (lower_wick < 0.3 * body) & 
                (close > open_) & (body > 0)).astype(int)
    
    def _detect_shooting_star(self, open_, high, low, close) -> pd.Series:
        """Detect shooting star."""
        body = (close - open_).abs()
        lower_wick = open_.where(close > open_, close) - low
        upper_wick = high - close.where(close > open_, open_)
        
        return ((upper_wick > 2 * body) & (lower_wick < 0.3 * body) & 
                (close < open_) & (body > 0)).astype(int)
    
    def _detect_doji(self, open_, high, low, close) -> pd.Series:
        """Detect doji."""
        body = (close - open_).abs()
        range_ = high - low
        return (body < range_ * 0.1).astype(int)
    
    def _detect_gravestone_doji(self, open_, high, low, close) -> pd.Series:
        """Detect gravestone doji."""
        body = (close - open_).abs()
        upper_wick = high - close.where(close > open_, open_)
        lower_wick = open_.where(close > open_, close) - low
        return ((body < (high - low) * 0.1) & (upper_wick > 3 * lower_wick)).astype(int)
    
    def _detect_dragonfly_doji(self, open_, high, low, close) -> pd.Series:
        """Detect dragonfly doji."""
        body = (close - open_).abs()
        upper_wick = high - close.where(close > open_, open_)
        lower_wick = open_.where(close > open_, close) - low
        return ((body < (high - low) * 0.1) & (lower_wick > 3 * upper_wick)).astype(int)
    
    def _detect_morning_star(self, open_, high, low, close) -> pd.Series:
        """Detect morning star (3-candle pattern)."""
        c1_open, c1_close = open_.shift(2), close.shift(2)
        c2_open, c2_close = open_.shift(1), close.shift(1)
        c3_open, c3_close = open_, close
        
        c1_bearish = c1_close < c1_open
        c2_small = (c2_close - c2_open).abs() < (c1_close - c1_open).abs() * 0.3
        c2_gap_down = c2_open < c1_close
        c3_bullish = c3_close > c3_open
        c3_strong = c3_close > (c1_open + c1_close) / 2
        
        return (c1_bearish & c2_small & c2_gap_down & c3_bullish & c3_strong).astype(int)
    
    def _detect_evening_star(self, open_, high, low, close) -> pd.Series:
        """Detect evening star (3-candle pattern)."""
        c1_open, c1_close = open_.shift(2), close.shift(2)
        c2_open, c2_close = open_.shift(1), close.shift(1)
        c3_open, c3_close = open_, close
        
        c1_bullish = c1_close > c1_open
        c2_small = (c2_close - c2_open).abs() < (c1_close - c1_open).abs() * 0.3
        c2_gap_up = c2_open > c1_close
        c3_bearish = c3_close < c3_open
        c3_strong = c3_close < (c1_open + c1_close) / 2
        
        return (c1_bullish & c2_small & c2_gap_up & c3_bearish & c3_strong).astype(int)
    
    def _detect_three_white_soldiers(self, open_, high, low, close) -> pd.Series:
        """Detect three white soldiers."""
        c1_bullish = close.shift(2) > open_.shift(2)
        c2_bullish = close.shift(1) > open_.shift(1)
        c3_bullish = close > open_
        
        c2_inside_c1 = (open_.shift(1) > open_.shift(2)) & (open_.shift(1) < close.shift(2))
        c3_inside_c2 = (open_ > open_.shift(1)) & (open_ < close.shift(1))
        
        return (c1_bullish & c2_bullish & c3_bullish & c2_inside_c1 & c3_inside_c2).astype(int)
    
    def _detect_three_black_crows(self, open_, high, low, close) -> pd.Series:
        """Detect three black crows."""
        c1_bearish = close.shift(2) < open_.shift(2)
        c2_bearish = close.shift(1) < open_.shift(1)
        c3_bearish = close < open_
        
        c2_inside_c1 = (open_.shift(1) < open_.shift(2)) & (open_.shift(1) > close.shift(2))
        c3_inside_c2 = (open_ < open_.shift(1)) & (open_ > close.shift(1))
        
        return (c1_bearish & c2_bearish & c3_bearish & c2_inside_c1 & c3_inside_c2).astype(int)
    
    def _detect_inside_bar(self, high, low) -> pd.Series:
        """Detect inside bar."""
        prev_high, prev_low = high.shift(1), low.shift(1)
        return ((high < prev_high) & (low > prev_low)).astype(int)
    
    def _detect_outside_bar(self, high, low) -> pd.Series:
        """Detect outside bar."""
        prev_high, prev_low = high.shift(1), low.shift(1)
        return ((high > prev_high) & (low < prev_low)).astype(int)
    
    def _detect_swing_high(self, high: pd.Series) -> pd.Series:
        """Detect swing highs."""
        lookback = self.swing_lookback
        swing_high = pd.Series(0, index=high.index)
        
        for i in range(lookback, len(high) - lookback):
            window = high.iloc[i-lookback:i+lookback+1]
            if high.iloc[i] == window.max():
                # Check minimum swing size
                if i > 0 and i < len(high) - 1:
                    left_max = high.iloc[max(0, i-lookback):i].max()
                    right_max = high.iloc[i+1:min(len(high), i+lookback+1)].max()
                    if (high.iloc[i] - left_max) / high.iloc[i] > self.min_swing_pct or \
                       (high.iloc[i] - right_max) / high.iloc[i] > self.min_swing_pct:
                        swing_high.iloc[i] = 1
        
        return swing_high
    
    def _detect_swing_low(self, low: pd.Series) -> pd.Series:
        """Detect swing lows."""
        lookback = self.swing_lookback
        swing_low = pd.Series(0, index=low.index)
        
        for i in range(lookback, len(low) - lookback):
            window = low.iloc[i-lookback:i+lookback+1]
            if low.iloc[i] == window.min():
                if i > 0 and i < len(low) - 1:
                    left_min = low.iloc[max(0, i-lookback):i].min()
                    right_min = low.iloc[i+1:min(len(low), i+lookback+1)].min()
                    if (left_min - low.iloc[i]) / low.iloc[i] > self.min_swing_pct or \
                       (right_min - low.iloc[i]) / low.iloc[i] > self.min_swing_pct:
                        swing_low.iloc[i] = 1
        
        return swing_low
    
    def _distance_to_swing(self, price: pd.Series, swing: pd.Series) -> pd.Series:
        """Calculate distance to nearest swing point."""
        swing_indices = swing[swing == 1].index
        if len(swing_indices) == 0:
            return pd.Series(np.nan, index=price.index)
        
        distances = pd.Series(np.nan, index=price.index)
        for idx in swing_indices:
            loc = price.index.get_loc(idx)
            # Forward fill distance
            for i in range(loc, len(price)):
                if pd.isna(distances.iloc[i]):
                    distances.iloc[i] = (price.iloc[i] - price.iloc[loc]) / price.iloc[loc]
                else:
                    break
        return distances.ffill()
    
    def _higher_high(self, high: pd.Series) -> pd.Series:
        """Detect higher high."""
        return (high > high.shift(1)) & (high.shift(1) > high.shift(2)).astype(int)
    
    def _higher_low(self, low: pd.Series) -> pd.Series:
        """Detect higher low."""
        return (low > low.shift(1)) & (low.shift(1) > low.shift(2)).astype(int)
    
    def _lower_high(self, high: pd.Series) -> pd.Series:
        """Detect lower high."""
        return (high < high.shift(1)) & (high.shift(1) < high.shift(2)).astype(int)
    
    def _lower_low(self, low: pd.Series) -> pd.Series:
        """Detect lower low."""
        return (low < low.shift(1)) & (low.shift(1) < low.shift(2)).astype(int)
    
    def _near_level(self, price: pd.Series, level_signal: pd.Series, threshold: float = 0.002) -> pd.Series:
        """Check if price is near a level."""
        levels = price[level_signal == 1]
        if len(levels) == 0:
            return pd.Series(0, index=price.index)
        
        near = pd.Series(0, index=price.index)
        for level_idx, level_price in levels.items():
            dist = (price - level_price).abs() / level_price
            near = near | (dist < threshold).astype(int)
        
        return near


if __name__ == "__main__":
    import yfinance as yf
    df = yf.download('GC=F', period='5d', interval='1h', progress=False)
    df.columns = [c.lower() for c in df.columns]
    
    pa = PriceActionFeatures()
    features = pa.transform(df)
    print(f"Price action features: {len(features.columns)}")
    print(features.tail())
