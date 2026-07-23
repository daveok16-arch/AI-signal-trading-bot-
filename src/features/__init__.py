"""Feature engineering module for XAUUSD Scalping System."""
from .engineer import FeatureEngineer, generate_training_data
from .price_action import PriceActionFeatures
from .market_structure import MarketStructureFeatures
from .volatility import VolatilityFeatures
from .momentum import MomentumFeatures
from .volume import VolumeFeatures
from .selection import FeatureSelector

__all__ = [
    'FeatureEngineer',
    'generate_training_data',
    'PriceActionFeatures',
    'MarketStructureFeatures',
    'VolatilityFeatures',
    'MomentumFeatures',
    'VolumeFeatures',
    'FeatureSelector',
]
