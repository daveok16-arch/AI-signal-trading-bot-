"""Configuration management for XAUUSD ScalpingAI Scalping System."""
import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from functools import lru_cache


class Config:
    """Configuration manager for XAUUSD Scalping System."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._find_config()
        self._config: Dict[str, Any] = {}
        self._load()
    
    def _find_config(self) -> str:
        """Find config file in standard locations."""
        paths = [
            Path(__file__).parent / "config.yaml",
            Path.cwd() / "config" / "config.yaml",
            Path.cwd() / "config.yaml",
            Path(__file__).parent.parent / "config" / "config.yaml",
        ]
        for path in paths:
            if path.exists():
                return str(path)
        raise FileNotFoundError("Config file not found")
    
    def _load(self) -> None:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            self._config = yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation key."""
        keys = key.split('.')
        value = self._config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
            if value is None:
                return default
        return value
    
    def __getitem__(self, key: str) -> Any:
        return self.get(key)
    
    def __getattr__(self, name: str) -> Any:
        if name in self._config:
            return self._config[name]
        raise AttributeError(f"'Config' object has no attribute '{name}'")
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section."""
        return self._config.get(section, {})
    
    def reload(self) -> None:
        """Reload configuration from file."""
        self._load()


@lru_cache(maxsize=1)
def get_config(config_path: Optional[str] = None) -> Config:
    """Get cached configuration instance."""
    return Config(config_path)


# Global config instance
config = get_config()
