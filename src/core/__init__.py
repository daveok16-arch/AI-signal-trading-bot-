"""Core application module for XAUUSD Scalping System."""
from .lifecycle import Application, ApplicationState, LifecycleManager, GracefulShutdown
from .config import AppSettings, load_config_from_env, TelegramSettings, APISettings, DashboardSettings

__all__ = [
    'Application', 
    'ApplicationState',
    'LifecycleManager', 
    'GracefulShutdown',
    'AppSettings',
    'load_config_from_env',
    'TelegramSettings',
    'APISettings',
    'DashboardSettings',
]
