"""Evaluation module for XAUUSD Scalping System."""
from .metrics import calculate_metrics, MetricsCalculator
from .reporting import generate_report

__all__ = ['calculate_metrics', 'MetricsCalculator', 'generate_report']
