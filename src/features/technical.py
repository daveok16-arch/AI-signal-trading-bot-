"""Technical indicators for XAUUSD Scalping System."""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging
from numba import jit

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Comprehensive technical indicator calculations."""
    
    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average."""
        return series.rolling(window=period, min_periods=1).mean()
    
    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average."""
        return series.ewm(span=period, adjust=False, min_periods=1).mean()
    
    @staticmethod
    def wma(series: pd.Series, period: int) -> pd.Series:
        """Weighted Moving Average."""
        weights = np.arange(1, period + 1)
        return series.rolling(window=period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(
        series: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> Dict[str, pd.Series]:
        """MACD with signal line and histogram."""
        ema_fast = TechnicalIndicators.ema(series, fast)
        ema_slow = TechnicalIndicators.ema(series, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram,
        }
    
    @staticmethod
    def bollinger_bands(
        series: pd.Series,
        period: int = 20,
        std: float = 2.0,
    ) -> Dict[str, pd.Series]:
        """Bollinger Bands."""
        sma = TechnicalIndicators.sma(series, period)
        std_dev = series.rolling(window=period).std()
        upper = sma + std * std_dev
        lower = sma - std * std_dev
        bandwidth = (upper - lower) / (sma + 1e-10)
        percent_b = (series - lower) / (upper - lower + 1e-10)
        return {
            'upper': upper,
            'middle': sma,
            'lower': lower,
            'bandwidth': bandwidth,
            'percent_b': percent_b,
        }
    
    @staticmethod
    def atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """Average True Range."""
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()
    
    @staticmethod
    def natr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """Normalized Average True Range."""
        atr_val = TechnicalIndicators.atr(high, low, close, period)
        return (atr_val / close) * 100
    
    @staticmethod
    def stoch(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3,
    ) -> Dict[str, pd.Series]:
        """Stochastic Oscillator."""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
        d = k.rolling(window=d_period).mean()
        return {'k': k, 'd': d}
    
    @staticmethod
    def williams_r(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """Williams %R."""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        return -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    @staticmethod
    def cci(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 20,
    ) -> pd.Series:
        """Commodity Channel Index."""
        tp = (high + low + close) / 3
        sma_tp = TechnicalIndicators.sma(tp, period)
        mad = tp.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
        )
        return (tp - sma_tp) / (0.015 * mad + 1e-10)
    
    @staticmethod
    def adx(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> Dict[str, pd.Series]:
        """Average Directional Index."""
        plus_dm = high.diff()
        minus_dm = low.diff().abs()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = TechnicalIndicators.atr(high, low, close, period=1)
        
        plus_di = 100 * (plus_dm.rolling(period).mean() / (tr.rolling(period).mean() + 1e-10))
        minus_di = 100 * (minus_dm.rolling(period).mean() / (tr.rolling(period).mean() + 1e-10))
        
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(period).mean()
        
        return {'adx': adx, 'plus_di': plus_di, 'minus_di': minus_di}
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume."""
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv
    
    @staticmethod
    def mfi(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """Money Flow Index."""
        tp = (high + low + close) / 3
        money_flow = tp * volume
        
        positive_flow = money_flow.where(tp > tp.shift(1), 0).rolling(period).sum()
        negative_flow = money_flow.where(tp < tp.shift(1), 0).rolling(period).sum()
        
        mfi = 100 - (100 / (1 + positive_flow / (negative_flow + 1e-10)))
        return mfi
    
    @staticmethod
    def vwap(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
    ) -> pd.Series:
        """Volume Weighted Average Price."""
        tp = (high + low + close) / 3
        return (tp * volume).cumsum() / volume.cumsum()
    
    @staticmethod
    def supertrend(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 10,
        multiplier: float = 3.0,
    ) -> Dict[str, pd.Series]:
        """Supertrend indicator."""
        atr_val = TechnicalIndicators.atr(high, low, close, period)
        
        basic_upper = (high + low) / 2 + multiplier * atr_val
        basic_lower = (high + low) / 2 - multiplier * atr_val
        
        final_upper = basic_upper.copy()
        final_lower = basic_lower.copy()
        supertrend = pd.Series(index=close.index, dtype=float)
        trend = pd.Series(index=close.index, dtype=int)
        
        for i in range(1, len(close)):
            # Final upper band
            if basic_upper.iloc[i] < final_upper.iloc[i-1] or close.iloc[i-1] > final_upper.iloc[i-1]:
                final_upper.iloc[i] = basic_upper.iloc[i]
            else:
                final_upper.iloc[i] = final_upper.iloc[i-1]
            
            # Final lower band
            if basic_lower.iloc[i] > final_lower.iloc[i-1] or close.iloc[i-1] < final_lower.iloc[i-1]:
                final_lower.iloc[i] = basic_lower.iloc[i]
            else:
                final_lower.iloc[i] = final_lower.iloc[i-1]
            
            # Supertrend
            if i == 1:
                supertrend.iloc[i] = final_upper.iloc[i] if close.iloc[i] <= final_upper.iloc[i] else final_lower.iloc[i]
                trend.iloc[i] = 1 if close.iloc[i] > final_upper.iloc[i] else -1
            else:
                if trend.iloc[i-1] == 1 and close.iloc[i] < final_lower.iloc[i]:
                    trend.iloc[i] = -1
                    supertrend.iloc[i] = final_upper.iloc[i]
                elif trend.iloc[i-1] == -1 and close.iloc[i] > final_upper.iloc[i]:
                    trend.iloc[i] = 1
                    supertrend.iloc[i] = final_lower.iloc[i]
                else:
                    trend.iloc[i] = trend.iloc[i-1]
                    supertrend.iloc[i] = final_lower.iloc[i] if trend.iloc[i] == 1 else final_upper.iloc[i]
        
        return {
            'supertrend': supertrend,
            'trend': trend,
            'upper': final_upper,
            'lower': final_lower,
        }
    
    @staticmethod
    def ichimoku(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        tenkan: int = 9,
        kijun: int = 26,
        senkou_b: int = 52,
    ) -> Dict[str, pd.Series]:
        """Ichimoku Cloud."""
        tenkan_sen = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
        kijun_sen = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
        senkou_span_b = ((high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2).shift(kijun)
        chikou_span = close.shift(-kijun)
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span,
        }
    
    @staticmethod
    def keltner_channels(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 20,
        atr_period: int = 10,
        multiplier: float = 2.0,
    ) -> Dict[str, pd.Series]:
        """Keltner Channels."""
        ema = TechnicalIndicators.ema(close, period)
        atr_val = TechnicalIndicators.atr(high, low, close, atr_period)
        
        upper = ema + multiplier * atr_val
        lower = ema - multiplier * atr_val
        
        return {'upper': upper, 'middle': ema, 'lower': lower}
    
    @staticmethod
    def donchian_channels(
        high: pd.Series,
        low: pd.Series,
        period: int = 20,
    ) -> Dict[str, pd.Series]:
        """Donchian Channels."""
        upper = high.rolling(period).max()
        lower = low.rolling(period).min()
        middle = (upper + lower) / 2
        return {'upper': upper, 'middle': middle, 'lower': lower}
    
    @staticmethod
    def pivot_points(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> Dict[str, pd.Series]:
        """Pivot Points (Standard)."""
        pp = (high + low + close) / 3
        r1 = 2 * pp - low
        s1 = 2 * pp - high
        r2 = pp + (high - low)
        s2 = pp - (high - low)
        r3 = high + 2 * (pp - low)
        s3 = low - 2 * (high - pp)
        
        return {
            'pp': pp, 'r1': r1, 's1': s1,
            'r2': r2, 's2': s2, 'r3': r3, 's3': s3,
        }
    
    @staticmethod
    def fibonacci_retracements(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        lookback: int = 50,
    ) -> Dict[str, pd.Series]:
        """Dynamic Fibonacci retracement levels."""
        highest_high = high.rolling(lookback).max()
        lowest_low = low.rolling(lookback).min()
        diff = highest_high - lowest_low
        
        levels = {
            'fib_0': highest_high,
            'fib_236': highest_high - 0.236 * diff,
            'fib_382': highest_high - 0.382 * diff,
            'fib_500': highest_high - 0.500 * diff,
            'fib_618': highest_high - 0.618 * diff,
            'fib_786': highest_high - 0.786 * diff,
            'fib_100': lowest_low,
        }
        return levels
    
    @staticmethod
    def parabolic_sar(
        high: pd.Series,
        low: pd.Series,
        af_start: float = 0.02,
        af_increment: float = 0.02,
        af_max: float = 0.2,
    ) -> pd.Series:
        """Parabolic SAR."""
        sar = pd.Series(index=high.index, dtype=float)
        trend = pd.Series(index=high.index, dtype=int)
        af = pd.Series(index=high.index, dtype=float)
        ep = pd.Series(index=high.index, dtype=float)
        
        # Initialize
        sar.iloc[0] = low.iloc[0]
        trend.iloc[0] = 1  # 1 = up, -1 = down
        af.iloc[0] = af_start
        ep.iloc[0] = high.iloc[0]
        
        for i in range(1, len(high)):
            if trend.iloc[i-1] == 1:  # Uptrend
                sar.iloc[i] = sar.iloc[i-1] + af.iloc[i-1] * (ep.iloc[i-1] - sar.iloc[i-1])
                sar.iloc[i] = min(sar.iloc[i], low.iloc[i-1], low.iloc[i-2] if i > 1 else low.iloc[i-1])
                
                if high.iloc[i] > ep.iloc[i-1]:
                    ep.iloc[i] = high.iloc[i]
                    af.iloc[i] = min(af.iloc[i-1] + af_increment, af_max)
                else:
                    ep.iloc[i] = ep.iloc[i-1]
                    af.iloc[i] = af.iloc[i-1]
                
                if low.iloc[i] < sar.iloc[i]:
                    trend.iloc[i] = -1
                    sar.iloc[i] = ep.iloc[i-1]
                    ep.iloc[i] = low.iloc[i]
                    af.iloc[i] = af_start
                else:
                    trend.iloc[i] = 1
                    
            else:  # Downtrend
                sar.iloc[i] = sar.iloc[i-1] - af.iloc[i-1] * (sar.iloc[i-1] - ep.iloc[i-1])
                sar.iloc[i] = max(sar.iloc[i], high.iloc[i-1], high.iloc[i-2] if i > 1 else high.iloc[i-1])
                
                if low.iloc[i] < ep.iloc[i-1]:
                    ep.iloc[i] = low.iloc[i]
                    af.iloc[i] = min(af.iloc[i-1] + af_increment, af_max)
                else:
                    ep.iloc[i] = ep.iloc[i-1]
                    af.iloc[i] = af.iloc[i-1]
                
                if high.iloc[i] > sar.iloc[i]:
                    trend.iloc[i] = 1
                    sar.iloc[i] = ep.iloc[i-1]
                    ep.iloc[i] = high.iloc[i]
                    af.iloc[i] = af_start
                else:
                    trend.iloc[i] = -1
        
        return sar
    
    @staticmethod
    def trix(series: pd.Series, period: int = 15) -> pd.Series:
        """TRIX indicator."""
        ema1 = TechnicalIndicators.ema(series, period)
        ema2 = TechnicalIndicators.ema(ema1, period)
        ema3 = TechnicalIndicators.ema(ema2, period)
        return ema3.pct_change() * 100
    
    @staticmethod
    def cmo(series: pd.Series, period: int = 14) -> pd.Series:
        """Chande Momentum Oscillator."""
        diff = series.diff()
        pos_sum = diff.where(diff > 0, 0).rolling(period).sum()
        neg_sum = (-diff.where(diff < 0, 0)).rolling(period).sum()
        return 100 * (pos_sum - neg_sum) / (pos_sum + neg_sum + 1e-10)
    
    @staticmethod
    def vhf(close: pd.Series, period: int = 28) -> pd.Series:
        """Vertical Horizontal Filter."""
        highest = close.rolling(period).max()
        lowest = close.rolling(period).min()
        numerator = (close - close.shift(period)).abs()
        denominator = (close.diff().abs()).rolling(period).sum()
        return numerator / (denominator + 1e-10)
    
    @staticmethod
    def all_indicators(
        df: pd.DataFrame,
        config: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """Calculate all technical indicators."""
        result = df.copy()
        
        high, low, close, volume = df['high'], df['low'], df['close'], df['volume']
        
        # Default config
        if config is None:
            config = {
                'sma': [5, 10, 20, 50, 100],
                'ema': [5, 10, 20, 50],
                'rsi': [14, 21],
                'macd': {'fast': 12, 'slow': 26, 'signal': 9},
                'bb': {'period': 20, 'std': 2},
                'atr': [14, 21],
                'stoch': {'k': 14, 'd': 3},
                'williams_r': [14],
                'cci': [20],
                'adx': [14],
                'obv': True,
                'mfi': [14],
                'vwap': True,
                'supertrend': {'period': 10, 'multiplier': 3},
                'ichimoku': {'tenkan': 9, 'kijun': 26, 'senkou_b': 52},
            }
        
        # SMAs
        for period in config.get('sma', []):
            result[f'sma_{period}'] = TechnicalIndicators.sma(close, period)
            result[f'sma_{period}_slope'] = result[f'sma_{period}'].diff()
            result[f'close_sma_{period}_ratio'] = close / result[f'sma_{period}']
        
        # EMAs
        for period in config.get('ema', []):
            result[f'ema_{period}'] = TechnicalIndicators.ema(close, period)
            result[f'ema_{period}_slope'] = result[f'ema_{period}'].diff()
            result[f'close_ema_{period}_ratio'] = close / result[f'ema_{period}']
        
        # RSI
        for period in config.get('rsi', []):
            result[f'rsi_{period}'] = TechnicalIndicators.rsi(close, period)
        
        # MACD
        macd_config = config.get('macd', {})
        macd = TechnicalIndicators.macd(
            close,
            macd_config.get('fast', 12),
            macd_config.get('slow', 26),
            macd_config.get('signal', 9),
        )
        result['macd'] = macd['macd']
        result['macd_signal'] = macd['signal']
        result['macd_hist'] = macd['histogram']
        
        # Bollinger Bands
        bb_config = config.get('bb', {})
        bb = TechnicalIndicators.bollinger_bands(
            close,
            bb_config.get('period', 20),
            bb_config.get('std', 2.0),
        )
        result['bb_upper'] = bb['upper']
        result['bb_middle'] = bb['middle']
        result['bb_lower'] = bb['lower']
        result['bb_bandwidth'] = bb['bandwidth']
        result['bb_percent_b'] = bb['percent_b']
        
        # ATR
        for period in config.get('atr', []):
            result[f'atr_{period}'] = TechnicalIndicators.atr(high, low, close, period)
            result[f'natr_{period}'] = TechnicalIndicators.natr(high, low, close, period)
        
        # Stochastic
        stoch_config = config.get('stoch', {})
        stoch = TechnicalIndicators.stoch(
            high, low, close,
            stoch_config.get('k', 14),
            stoch_config.get('d', 3),
        )
        result['stoch_k'] = stoch['k']
        result['stoch_d'] = stoch['d']
        
        # Williams %R
        for period in config.get('williams_r', []):
            result[f'williams_r_{period}'] = TechnicalIndicators.williams_r(high, low, close, period)
        
        # CCI
        for period in config.get('cci', []):
            result[f'cci_{period}'] = TechnicalIndicators.cci(high, low, close, period)
        
        # ADX
        for period in config.get('adx', []):
            adx = TechnicalIndicators.adx(high, low, close, period)
            result[f'adx_{period}'] = adx['adx']
            result[f'plus_di_{period}'] = adx['plus_di']
            result[f'minus_di_{period}'] = adx['minus_di']
        
        # OBV
        if config.get('obv'):
            result['obv'] = TechnicalIndicators.obv(close, volume)
            result['obv_slope'] = result['obv'].diff()
        
        # MFI
        for period in config.get('mfi', []):
            result[f'mfi_{period}'] = TechnicalIndicators.mfi(high, low, close, volume, period)
        
        # VWAP
        if config.get('vwap'):
            result['vwap'] = TechnicalIndicators.vwap(high, low, close, volume)
            result['close_vwap_ratio'] = close / result['vwap']
        
        # Supertrend
        st_config = config.get('supertrend', {})
        st = TechnicalIndicators.supertrend(
            high, low, close,
            st_config.get('period', 10),
            st_config.get('multiplier', 3.0),
        )
        result['supertrend'] = st['supertrend']
        result['supertrend_trend'] = st['trend']
        
        # Ichimoku
        ichi_config = config.get('ichimoku', {})
        ichi = TechnicalIndicators.ichimoku(
            high, low, close,
            ichi_config.get('tenkan', 9),
            ichi_config.get('kijun', 26),
            ichi_config.get('senkou_b', 52),
        )
        result['ichimoku_tenkan'] = ichi['tenkan_sen']
        result['ichimoku_kijun'] = ichi['kijun_sen']
        result['ichimoku_span_a'] = ichi['senkou_span_a']
        result['ichimoku_span_b'] = ichi['senkou_span_b']
        
        return result


if __name__ == "__main__":
    # Test indicators
    import yfinance as yf
    df = yf.download('GC=F', period='1mo', interval='1h', progress=False)
    df.columns = [c.lower() for c in df.columns]
    
    tech = TechnicalIndicators()
    result = tech.all_indicators(df)
    print(f"Columns: {len(result.columns)}")
    print(result.tail())
