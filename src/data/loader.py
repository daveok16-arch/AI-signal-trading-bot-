"""Data loader module for XAUUSD Scalping System."""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Union
import logging
from pathlib import Path
import time

from .yahoo_client import YahooFinanceClient, MultiTickerClient
from .cache import DataCache, get_data_cache
from ..config import get_config

logger = logging.getLogger(__name__)


class DataLoader:
    """High-level data loader with caching and preprocessing."""
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        cache: Optional[DataCache] = None,
    ):
        self.config = get_config(config_path)
        self.cache = cache or get_data_cache()
        
        # Initialize clients
        data_config = self.config.get_section('data')
        self.yahoo_client = YahooFinanceClient(
            ticker=data_config.get('symbol', 'GC=F'),
            cache_dir=data_config.get('cache_dir', 'data/cache'),
            max_retries=data_config.get('max_retries', 3),
            retry_delay=data_config.get('retry_delay', 5),
        )
        self.multi_client = MultiTickerClient(
            cache_dir=data_config.get('cache_dir', 'data/cache')
        )
        
        # Data config
        self.default_interval = data_config.get('interval', '1m')
        self.default_period = data_config.get('period', '7d')
        self.timezone = data_config.get('timezone', 'UTC')
    
    def load_ohlcv(
        self,
        interval: Optional[str] = None,
        period: Optional[str] = None,
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None,
        use_cache: bool = True,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Load OHLCV data with caching and fallback.
        
        Args:
            interval: Data interval
            period: Data period
            start: Start date
            end: End date
            use_cache: Whether to use cache
            force_refresh: Force refresh from source
            
        Returns:
            OHLCV DataFrame
        """
        interval = interval or self.default_interval
        period = period or self.default_period
        
        # Generate cache key
        cache_key = self._generate_cache_key(interval, period, start, end)
        
        # Try cache first (even if use_cache is False, try to get old cache on network failure)
        cached = self.cache.get_dataframe(cache_key) if use_cache else None
        if cached is not None and not force_refresh:
            logger.info(f"Loaded {len(cached)} rows from cache")
            return cached
        
        # Try to fetch from source
        try:
            logger.info(f"Fetching {interval} data for {period} from Yahoo Finance")
            df = self.yahoo_client.fetch(
                interval=interval,
                period=period,
                start=start,
                end=end,
                use_cache=False,  # We handle caching ourselves
            )
            
            if df.empty:
                logger.warning("Yahoo Finance returned empty data")
                # Fallback to stale cache
                if cached is not None:
                    logger.info("Falling back to cached data")
                    return cached
                return pd.DataFrame()
            
            # Post-process
            df = self._post_process(df)
            
            # Cache result
            if use_cache and not df.empty:
                self.cache.set(cache_key, df)
            
            logger.info(f"Loaded {len(df)} rows from source")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch data from Yahoo Finance: {e}")
            # Fallback to cached data on network error
            if cached is not None:
                logger.warning("Network error - falling back to cached data")
                return cached
            logger.error("No cached data available - returning empty DataFrame")
            return pd.DataFrame()
    
    def _generate_cache_key(
        self,
        interval: str,
        period: str,
        start: Optional[Union[str, datetime]],
        end: Optional[Union[str, datetime]],
    ) -> str:
        """Generate cache key for data request."""
        if start and end:
            return f"ohlcv_{self.yahoo_client.ticker}_{interval}_{start}_{end}"
        return f"ohlcv_{self.yahoo_client.ticker}_{interval}_{period}"
    
    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Post-process OHLCV data."""
        df = df.copy()
        
        # Ensure timezone
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        elif df.index.tz != self.timezone:
            df.index = df.index.tz_convert(self.timezone)
        
        # Add derived columns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['hl_spread'] = (df['high'] - df['low']) / df['close']
        df['oc_spread'] = (df['close'] - df['open']) / df['open']
        
        # Volume metrics
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Price position in range
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
        
        return df
    
    def load_multi_timeframe(
        self,
        intervals: List[str] = ['1m', '5m', '15m', '1h'],
        period: str = '1d',
        use_cache: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Load data for multiple timeframes."""
        results = {}
        for interval in intervals:
            try:
                results[interval] = self.load_ohlcv(
                    interval=interval,
                    period=period,
                    use_cache=use_cache,
                )
            except Exception as e:
                logger.error(f"Failed to load {interval}: {e}")
                results[interval] = pd.DataFrame()
        return results
    
    def load_with_features(
        self,
        interval: str = '1m',
        period: str = '1d',
        feature_version: str = 'v1',
        use_cache: bool = True,
    ) -> tuple:
        """Load OHLCV data with pre-computed features."""
        from ..features import FeatureEngineer
        
        # Load raw data
        df = self.load_ohlcv(interval=interval, period=period, use_cache=use_cache)
        
        if df.empty:
            return df, pd.DataFrame()
        
        # Generate features
        engineer = FeatureEngineer(config=self.config)
        features = engineer.transform(df)
        
        return df, features
    
    def get_latest_data(
        self,
        interval: str = '1m',
        n_bars: int = 100,
    ) -> pd.DataFrame:
        """Get latest N bars of data."""
        end = datetime.now()
        start = end - timedelta(minutes=n_bars * self._interval_to_minutes(interval))
        
        df = self.load_ohlcv(interval=interval, start=start, end=end, use_cache=False)
        return df.tail(n_bars)
    
    def _interval_to_minutes(self, interval: str) -> int:
        """Convert interval string to minutes."""
        mapping = {
            '1m': 1, '2m': 2, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '4h': 240, '1d': 1440, '1wk': 10080, '1mo': 43200,
        }
        return mapping.get(interval, 1)
    
    def check_connection(self) -> Dict[str, Any]:
        """
        Check connection to Yahoo Finance and return status.
        
        Returns:
            Dictionary with connection status details
        """
        result = {
            "connected": False,
            "latency_ms": None,
            "data_available": False,
            "error": None,
            "timestamp": datetime.now().isoformat(),
        }
        
        start_time = time.time()
        try:
            # Try to fetch minimal data
            df = self.yahoo_client.fetch(interval="1m", period="1d", use_cache=False)
            result["latency_ms"] = int((time.time() - start_time) * 1000)
            
            if df is not None and not df.empty:
                result["connected"] = True
                result["data_available"] = True
                result["bars"] = len(df)
                result["last_bar"] = df.index[-1].isoformat() if len(df) > 0 else None
            else:
                result["error"] = "No data returned"
                
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Connection check failed: {e}")
        
        return result
    
    def get_market_context(self) -> Dict[str, Any]:
        """Get current market context."""
        info = self.yahoo_client.get_ticker_info()
        current_price = self.yahoo_client.get_current_price()
        
        # Get correlation data
        correlations = self.multi_client.get_correlations()
        
        return {
            'ticker_info': info,
            'current_price': current_price,
            'correlations': correlations.to_dict() if not correlations.empty else {},
            'timestamp': datetime.now().isoformat(),
        }
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality."""
        if df.empty:
            return {'valid': False, 'reason': 'Empty DataFrame'}
        
        issues = []
        
        # Check for gaps
        expected_freq = pd.infer_freq(df.index)
        if expected_freq:
            expected_index = pd.date_range(
                df.index[0], df.index[-1], freq=expected_freq
            )
            missing = expected_index.difference(df.index)
            if len(missing) > 0:
                issues.append(f"Missing {len(missing)} bars ({len(missing)/len(expected_index)*100:.1f}%)")
        
        # Check for NaN
        nan_count = df[['open', 'high', 'low', 'close']].isna().sum().sum()
        if nan_count > 0:
            issues.append(f"NaN values in OHLC: {nan_count}")
        
        # Check for zero volume
        zero_vol = (df['volume'] == 0).sum()
        if zero_vol > len(df) * 0.1:
            issues.append(f"High zero volume bars: {zero_vol} ({zero_vol/len(df)*100:.1f}%)")
        
        # Check OHLC validity
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        ).sum()
        if invalid_ohlc > 0:
            issues.append(f"Invalid OHLC relationships: {invalid_ohlc}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'rows': len(df),
            'date_range': f"{df.index[0]} to {df.index[-1]}",
            'frequency': expected_freq,
        }


# Convenience functions
def load_gold_data(
    interval: str = '1m',
    period: str = '1d',
    **kwargs
) -> pd.DataFrame:
    """Convenience function to load XAUUSD data."""
    loader = DataLoader()
    return loader.load_ohlcv(interval=interval, period=period, **kwargs)


def load_multi_tf_data(
    intervals: List[str] = ['1m', '5m', '15m', '1h'],
    period: str = '1d',
) -> Dict[str, pd.DataFrame]:
    """Convenience function to load multi-timeframe data."""
    loader = DataLoader()
    return loader.load_multi_timeframe(intervals=intervals, period=period)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    loader = DataLoader()
    
    # Test loading
    df = loader.load_ohlcv(interval='1m', period='1d')
    print(f"Loaded {len(df)} rows")
    print(df.head())
    
    # Validate
    quality = loader.validate_data_quality(df)
    print(f"Data quality: {quality}")
