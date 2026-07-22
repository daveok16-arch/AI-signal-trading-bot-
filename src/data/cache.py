"""Data caching module for XAUUSD Scalping System."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
import pickle
import hashlib
import logging
from datetime import datetime, timedelta
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class DataCache:
    """Thread-safe data cache with multiple backend support."""
    
    def __init__(
        self,
        cache_dir: str = "data/cache",
        default_ttl_hours: int = 24,
        max_size_gb: float = 1.0,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = timedelta(hours=default_ttl_hours)
        self.max_size_bytes = max_size_gb * 1024 ** 3
        self._lock = threading.RLock()
        self._memory_cache: Dict[str, Any] = {}
        self._metadata: Dict[str, Dict] = {}
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        """Load cache metadata."""
        meta_file = self.cache_dir / "metadata.pkl"
        if meta_file.exists():
            try:
                with open(meta_file, 'rb') as f:
                    self._metadata = pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
                self._metadata = {}
    
    def _save_metadata(self) -> None:
        """Save cache metadata."""
        meta_file = self.cache_dir / "metadata.pkl"
        try:
            with open(meta_file, 'wb') as f:
                pickle.dump(self._metadata, f)
        except Exception as e:
            logger.warning(f"Failed to save cache metadata: {e}")
    
    def _get_key_hash(self, key: str) -> str:
        """Generate hash for cache key."""
        return hashlib.sha256(key.encode()).hexdigest()[:16]
    
    def _get_file_path(self, key: str, format: str = 'parquet') -> Path:
        """Get cache file path."""
        key_hash = self._get_key_hash(key)
        return self.cache_dir / f"{key_hash}.{format}"
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        if key not in self._metadata:
            return True
        expiry = self._metadata[key].get('expiry')
        if expiry is None:
            return False
        return datetime.now() > expiry
    
    def _get_size(self) -> int:
        """Get total cache size in bytes."""
        total = 0
        for meta in self._metadata.values():
            total += meta.get('size', 0)
        return total
    
    def _evict_oldest(self) -> None:
        """Evict oldest cache entries to make space."""
        if not self._metadata:
            return
        
        # Sort by last access time
        sorted_keys = sorted(
            self._metadata.keys(),
            key=lambda k: self._metadata[k].get('last_access', datetime.min)
        )
        
        # Remove oldest until under limit
        target_size = self.max_size_bytes * 0.8
        current_size = self._get_size()
        
        for key in sorted_keys:
            if current_size <= target_size:
                break
            self._remove_entry(key)
            current_size = self._get_size()
    
    def _remove_entry(self, key: str) -> None:
        """Remove cache entry."""
        # Remove files
        for fmt in ['parquet', 'pkl', 'npz']:
            path = self._get_file_path(key, fmt)
            if path.exists():
                path.unlink()
        
        # Remove from metadata
        if key in self._metadata:
            del self._metadata[key]
        
        # Remove from memory cache
        if key in self._memory_cache:
            del self._memory_cache[key]
    
    @contextmanager
    def _locked(self):
        """Thread-safe lock context."""
        self._lock.acquire()
        try:
            yield
        finally:
            self._lock.release()
    
    def set(
        self,
        key: str,
        data: Any,
        ttl: Optional[timedelta] = None,
        format: str = 'parquet',
    ) -> bool:
        """Store data in cache."""
        with self._locked():
            try:
                # Check size limit
                if self._get_size() > self.max_size_bytes * 0.9:
                    self._evict_oldest()
                
                file_path = self._get_file_path(key, format)
                
                # Save based on format
                if format == 'parquet' and isinstance(data, pd.DataFrame):
                    data.to_parquet(file_path, compression='snappy')
                elif format == 'pkl':
                    with open(file_path, 'wb') as f:
                        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                elif format == 'npz' and isinstance(data, dict):
                    np.savez_compressed(file_path, **data)
                else:
                    with open(file_path, 'wb') as f:
                        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                # Update metadata
                size = file_path.stat().st_size
                expiry = datetime.now() + (ttl or self.default_ttl)
                
                self._metadata[key] = {
                    'file': file_path.name,
                    'format': format,
                    'size': size,
                    'created': datetime.now(),
                    'expiry': expiry,
                    'last_access': datetime.now(),
                    'shape': getattr(data, 'shape', None),
                }
                
                self._save_metadata()
                return True
                
            except Exception as e:
                logger.error(f"Failed to cache data for key {key}: {e}")
                return False
    
    def get(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        """Retrieve data from cache."""
        with self._locked():
            if key not in self._metadata:
                return default
            
            if self._is_expired(key):
                self._remove_entry(key)
                return default
            
            meta = self._metadata[key]
            file_path = self.cache_dir / meta['file']
            
            if not file_path.exists():
                self._remove_entry(key)
                return default
            
            try:
                # Load based on format
                fmt = meta['format']
                if fmt == 'parquet':
                    data = pd.read_parquet(file_path)
                elif fmt == 'pkl':
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                elif fmt == 'npz':
                    data = dict(np.load(file_path))
                else:
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                
                # Update access time
                meta['last_access'] = datetime.now()
                self._save_metadata()
                
                return data
                
            except Exception as e:
                logger.error(f"Failed to load cache for key {key}: {e}")
                self._remove_entry(key)
                return default
    
    def get_dataframe(
        self,
        key: str,
        default: Optional[pd.DataFrame] = None,
    ) -> Optional[pd.DataFrame]:
        """Retrieve DataFrame from cache."""
        data = self.get(key, default)
        if isinstance(data, pd.DataFrame):
            return data
        return default
    
    def delete(self, key: str) -> bool:
        """Delete cache entry."""
        with self._locked():
            if key in self._metadata:
                self._remove_entry(key)
                self._save_metadata()
                return True
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        with self._locked():
            if key not in self._metadata:
                return False
            if self._is_expired(key):
                self._remove_entry(key)
                return False
            return True
    
    def clear(self, older_than: Optional[timedelta] = None) -> int:
        """Clear cache entries."""
        with self._locked():
            count = 0
            keys_to_remove = []
            
            for key, meta in self._metadata.items():
                if older_than:
                    created = meta.get('created')
                    if created and datetime.now() - created > older_than:
                        keys_to_remove.append(key)
                else:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                self._remove_entry(key)
                count += 1
            
            self._save_metadata()
            return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._locked():
            total_size = self._get_size()
            return {
                'entries': len(self._metadata),
                'total_size_mb': total_size / (1024 ** 2),
                'max_size_mb': self.max_size_bytes / (1024 ** 2),
                'usage_percent': (total_size / self.max_size_bytes) * 100,
                'oldest_entry': min(
                    (m['created'] for m in self._metadata.values()),
                    default=None
                ),
                'newest_entry': max(
                    (m['created'] for m in self._metadata.values()),
                    default=None
                ),
            }
    
    def list_keys(self, pattern: str = "*") -> List[str]:
        """List cache keys matching pattern."""
        import fnmatch
        with self._locked():
            return [k for k in self._metadata.keys() if fnmatch.fnmatch(k, pattern)]


class FeatureCache(DataCache):
    """Specialized cache for feature matrices."""
    
    def __init__(self, cache_dir: str = "data/cache/features"):
        super().__init__(cache_dir, default_ttl_hours=168, max_size_gb=2.0)  # 1 week
    
    def save_features(
        self,
        symbol: str,
        timeframe: str,
        features: pd.DataFrame,
        feature_version: str = "v1",
    ) -> bool:
        """Save feature matrix."""
        key = f"features_{symbol}_{timeframe}_{feature_version}"
        return self.set(key, features, format='parquet')
    
    def load_features(
        self,
        symbol: str,
        timeframe: str,
        feature_version: str = "v1",
    ) -> Optional[pd.DataFrame]:
        """Load feature matrix."""
        key = f"features_{symbol}_{timeframe}_{feature_version}"
        return self.get_dataframe(key)


class ModelCache(DataCache):
    """Specialized cache for trained models."""
    
    def __init__(self, cache_dir: str = "models/cache"):
        super().__init__(cache_dir, default_ttl_hours=720, max_size_gb=5.0)  # 30 days
    
    def save_model(
        self,
        model_name: str,
        model: Any,
        metadata: Dict[str, Any],
        version: str = "v1",
    ) -> bool:
        """Save model with metadata."""
        key = f"model_{model_name}_{version}"
        data = {'model': model, 'metadata': metadata}
        return self.set(key, data, format='pkl')
    
    def load_model(
        self,
        model_name: str,
        version: str = "v1",
    ) -> tuple:
        """Load model and metadata."""
        key = f"model_{model_name}_{version}"
        data = self.get(key)
        if data:
            return data['model'], data['metadata']
        return None, None


# Global cache instances
_data_cache: Optional[DataCache] = None
_feature_cache: Optional[FeatureCache] = None
_model_cache: Optional[ModelCache] = None


def get_data_cache() -> DataCache:
    """Get global data cache instance."""
    global _data_cache
    if _data_cache is None:
        _data_cache = DataCache()
    return _data_cache


def get_feature_cache() -> FeatureCache:
    """Get global feature cache instance."""
    global _feature_cache
    if _feature_cache is None:
        _feature_cache = FeatureCache()
    return _feature_cache


def get_model_cache() -> ModelCache:
    """Get global model cache instance."""
    global _model_cache
    if _model_cache is None:
        _model_cache = ModelCache()
    return _model_cache


if __name__ == "__main__":
    # Test cache
    logging.basicConfig(level=logging.INFO)
    
    cache = DataCache("data/test_cache")
    
    # Test DataFrame caching
    df = pd.DataFrame({
        'a': np.random.randn(100),
        'b': np.random.randn(100),
    }, index=pd.date_range('2024-01-01', periods=100, freq='1min'))
    
    cache.set('test_df', df)
    loaded = cache.get_dataframe('test_df')
    print(f"Original shape: {df.shape}, Loaded shape: {loaded.shape}")
    print(f"Cache stats: {cache.get_stats()}")
    
    # Cleanup
    cache.clear()
