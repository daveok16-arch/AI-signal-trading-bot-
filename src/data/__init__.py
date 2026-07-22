"""Data acquisition module for XAUUSD Scalping System."""
from .yahoo_client import YahooFinanceClient
from .cache import DataCache
from .loader import DataLoader

__all__ = ['YahooFinanceClient', 'DataCache', 'DataLoader']
