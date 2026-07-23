"""Yahoo Finance data client for XAUUSD market data."""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union
import logging
from pathlib import Path
import time
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

logger = logging.getLogger(__name__)


class YahooFinanceClient:
    """Yahoo Finance client for XAUUSD market data."""
    
    # GC=F is Gold Futures Continuous Contract (recommended)
    XAUUSD_TICKER = "GC=F"
    # XAUUSD=X is the spot forex pair (may have limited data)
    XAUUSD_SPOT = "XAUUSD=X"
    
    INTERVALS = {
        '1m': '1m',
        '2m': '2m',
        '5m': '5m',
        '15m': '15m',
        '30m': '30m',
        '1h': '1h',
        '1d': '1d',
        '1wk': '1wk',
        '1mo': '1mo',
    }
    
    PERIODS = {
        '1d': '1d',
        '5d': '5d',
        '1mo': '1mo',
        '3mo': '3mo',
        '6mo': '6mo',
        '1y': '1y',
        '2y': '2y',
        '5y': '5y',
        '10y': '10y',
        'ytd': 'ytd',
        'max': 'max',
    }
    
    def __init__(
        self,
        ticker: str = XAUUSD_TICKER,
        cache_dir: str = "data/cache",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: int = 30,
    ):
        self.ticker = ticker
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self._ticker_obj = None
        self._session = None
        
    @property
    def ticker_obj(self) -> yf.Ticker:
        """Lazy load yfinance ticker object."""
        if self._ticker_obj is None:
            self._ticker_obj = yf.Ticker(self.ticker)
        return self._ticker_obj
    
    def _get_cache_path(self, interval: str, period: str) -> Path:
        """Generate cache file path."""
        safe_ticker = self.ticker.replace("=", "_").replace("=", "_")
        filename = f"{safe_ticker}_{interval}_{period}.parquet"
        return self.cache_dir / filename
    
    def _is_cache_valid(self, cache_path: Path, max_age_hours: int = 1) -> bool:
        """Check if cache file is valid and not too old."""
        if not cache_path.exists():
            return False
        age = datetime.now().timestamp() - cache_path.stat().st_mtime
        return age < (max_age_hours * 3600)
    
    def _save_to_cache(self, df: pd.DataFrame, cache_path: Path) -> None:
        """Save DataFrame to parquet cache."""
        try:
            df.to_parquet(cache_path, compression='snappy')
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _load_from_cache(self, cache_path: Path) -> Optional[pd.DataFrame]:
        """Load DataFrame from parquet cache."""
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None
    
    def fetch(
        self,
        interval: str = '1m',
        period: str = '1d',
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None,
        use_cache: bool = True,
        cache_max_age_hours: int = 1,
        prepost: bool = False,
        repair: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from Yahoo Finance.
        
        Args:
            interval: Data interval (1m, 5m, 15m, 1h, 1d, etc.)
            period: Period to fetch (1d, 5d, 1mo, 3mo, 1y, etc.)
            start: Start date (overrides period if provided)
            end: End date (overrides period if provided)
            use_cache: Whether to use cached data
            cache_max_age_hours: Maximum cache age in hours
            prepost: Include pre/post market data
            repair: Attempt to repair data issues
            
        Returns:
            DataFrame with OHLCV data
        """
        # Validate interval
        if interval not in self.INTERVALS:
            raise ValueError(f"Invalid interval: {interval}. Valid: {list(self.INTERVALS.keys())}")
        
        interval = self.INTERVALS[interval]
        
        # Handle period/start/end
        if start is not None or end is not None:
            period = None
            if isinstance(start, str):
                start = pd.Timestamp(start)
            if isinstance(end, str):
                end = pd.Timestamp(end)
        else:
            if period not in self.PERIODS:
                raise ValueError(f"Invalid period: {period}. Valid: {list(self.PERIODS.keys())}")
            period = self.PERIODS[period]
        
        # Check cache
        cache_path = self._get_cache_path(interval, period or f"{start}_{end}")
        if use_cache and self._is_cache_valid(cache_path, cache_max_age_hours):
            cached = self._load_from_cache(cache_path)
            if cached is not None:
                logger.info(f"Loaded data from cache: {cache_path}")
                return cached
        
        # Fetch from Yahoo Finance
        df = self._fetch_with_retry(interval, period, start, end, prepost, repair)
        
        # Clean and validate data
        df = self._clean_data(df)
        
        # Save to cache
        if use_cache and df is not None and not df.empty:
            self._save_to_cache(df, cache_path)
        
        return df
    
    def _fetch_with_retry(
        self,
        interval: str,
        period: Optional[str],
        start: Optional[datetime],
        end: Optional[datetime],
        prepost: bool,
        repair: bool,
    ) -> pd.DataFrame:
        """Fetch data with retry logic."""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                if period:
                    df = self.ticker_obj.history(
                        period=period,
                        interval=interval,
                        prepost=prepost,
                        repair=repair,
                        timeout=self.timeout,
                    )
                else:
                    df = self.ticker_obj.history(
                        start=start,
                        end=end,
                        interval=interval,
                        prepost=prepost,
                        repair=repair,
                        timeout=self.timeout,
                    )
                
                if df is not None and not df.empty:
                    return df
                    
            except Exception as e:
                last_error = e
                logger.warning(f"Fetch attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
        
        raise RuntimeError(f"Failed to fetch data after {self.max_retries} attempts: {last_error}")
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate OHLCV data."""
        if df is None or df.empty:
            return pd.DataFrame()
        
        # Make copy
        df = df.copy()
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Sort by index
        df = df.sort_index()
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        # Standardize column names
        df.columns = [c.lower().replace(' ', '_') for c in df.columns]
        
        # Ensure required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                if col == 'volume' and 'volume' not in df.columns:
                    df['volume'] = 0
                else:
                    raise ValueError(f"Missing required column: {col}")
        
        # Remove rows with NaN in OHLC
        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        
        # Forward fill volume NaN
        df['volume'] = df['volume'].ffill().fillna(0)
        
        # Validate OHLC relationships
        invalid = (df['high'] < df['low']) | (df['high'] < df['open']) | \
                  (df['high'] < df['close']) | (df['low'] > df['open']) | \
                  (df['low'] > df['close'])
        if invalid.any():
            logger.warning(f"Found {invalid.sum()} rows with invalid OHLC, removing")
            df = df[~invalid]
        
        # Remove zero volume rows (optional - keep for now)
        
        return df
    
    def fetch_multi_interval(
        self,
        intervals: List[str] = ['1m', '5m', '15m', '1h'],
        period: str = '1d',
        use_cache: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple intervals."""
        results = {}
        for interval in intervals:
            try:
                results[interval] = self.fetch(interval=interval, period=period, use_cache=use_cache)
            except Exception as e:
                logger.error(f"Failed to fetch {interval}: {e}")
                results[interval] = pd.DataFrame()
        return results
    
    def fetch_intraday(
        self,
        date: Union[str, datetime],
        interval: str = '1m',
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch intraday data for a specific date."""
        if isinstance(date, str):
            date = pd.Timestamp(date)
        
        start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        
        return self.fetch(
            interval=interval,
            start=start,
            end=end,
            use_cache=use_cache,
        )
    
    def get_ticker_info(self) -> Dict[str, Any]:
        """Get ticker metadata."""
        try:
            return self.ticker_obj.info
        except Exception as e:
            logger.warning(f"Failed to get ticker info: {e}")
            return {}
    
    def get_current_price(self) -> Optional[float]:
        """Get current market price."""
        try:
            info = self.ticker_obj.info
            return info.get('regularMarketPrice') or info.get('currentPrice')
        except Exception as e:
            logger.warning(f"Failed to get current price: {e}")
            return None
    
    def get_market_hours(self) -> Dict[str, Any]:
        """Get market hours info."""
        try:
            return self.ticker_obj.info.get('marketHours', {})
        except Exception:
            return {}
    
    def clear_cache(self, older_than_hours: Optional[int] = None) -> int:
        """Clear cache files."""
        count = 0
        for cache_file in self.cache_dir.glob("*.parquet"):
            if older_than_hours:
                age = datetime.now().timestamp() - cache_file.stat().st_mtime
                if age < older_than_hours * 3600:
                    continue
            cache_file.unlink()
            count += 1
        return count


class MultiTickerClient:
    """Client for fetching multiple related tickers."""
    
    RELATED_TICKERS = {
        'XAUUSD': 'XAUUSD=X',
        'GOLD_FUTURES': 'GC=F',
        'GOLD_FUTURES_2': 'GCM25.CMX',
        'SILVER': 'SI=F',
        'DXY': 'DX=F',
        'US10Y': '^TNX',
        'SP500': '^GSPC',
        'VIX': '^VIX',
        'USOIL': 'CL=F',
        'EURUSD': 'EURUSD=X',
        'USDJPY': 'USDJPY=X',
    }
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.clients = {
            name: YahooFinanceClient(ticker, cache_dir)
            for name, ticker in self.RELATED_TICKERS.items()
        }
    
    def fetch_all(
        self,
        interval: str = '1h',
        period: str = '1d',
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for all related tickers."""
        results = {}
        for name, client in self.clients.items():
            try:
                results[name] = client.fetch(interval=interval, period=period)
            except Exception as e:
                logger.error(f"Failed to fetch {name}: {e}")
                results[name] = pd.DataFrame()
        return results
    
    def get_correlations(
        self,
        interval: str = '1h',
        period: str = '1mo',
        base: str = 'XAUUSD',
    ) -> pd.Series:
        """Calculate correlations with base ticker."""
        data = self.fetch_all(interval=interval, period=period)
        base_data = data.get(base)
        if base_data is None or base_data.empty:
            return pd.Series()
        
        base_returns = base_data['close'].pct_change().dropna()
        correlations = {}
        
        for name, df in data.items():
            if name == base or df.empty:
                continue
            returns = df['close'].pct_change().dropna()
            # Align indices
            common_idx = base_returns.index.intersection(returns.index)
            if len(common_idx) > 10:
                corr = base_returns.loc[common_idx].corr(returns.loc[common_idx])
                correlations[name] = corr
        
        return pd.Series(correlations).sort_values(ascending=False)


# Convenience function
def get_gold_data(
    interval: str = '1m',
    period: str = '1d',
    **kwargs
) -> pd.DataFrame:
    """Convenience function to fetch XAUUSD data."""
    client = YahooFinanceClient()
    return client.fetch(interval=interval, period=period, **kwargs)


if __name__ == "__main__":
    # Test the client
    logging.basicConfig(level=logging.INFO)
    
    client = YahooFinanceClient()
    
    # Test fetch
    df = client.fetch(interval='1m', period='1d')
    print(f"Fetched {len(df)} rows")
    print(df.head())
    print(df.tail())
    
    # Test current price
    price = client.get_current_price()
    print(f"Current price: {price}")
