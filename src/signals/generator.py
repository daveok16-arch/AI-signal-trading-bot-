"""Signal generation for XAUUSD Scalping System."""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, field
import logging

from ..config import get_config

logger = logging.getLogger(__name__)


class SignalType(Enum):
    BUY = 1
    SELL = -1
    WAIT = 0


@dataclass
class Signal:
    timestamp: pd.Timestamp
    signal_type: SignalType
    confidence: float
    price: float
    features: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SignalGenerator:
    """Generate trading signals from model predictions."""
    
    def __init__(
        self,
        model: Any,
        config_path: Optional[str] = None,
        scaler: Any = None,
        feature_names: Optional[List[str]] = None,
    ):
        self.config = get_config(config_path)
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        
        sig_config = self.config.get_section('signals')
        self.confidence_threshold = sig_config.get('confidence_threshold', 0.55)
        self.min_risk_reward = sig_config.get('min_risk_reward', 1.5)
        self.consensus_threshold = sig_config.get('consensus_threshold', 0.6)
        self.voting_method = sig_config.get('voting_method', 'weighted')
        self.signal_smoothing = sig_config.get('signal_smoothing', 3)
        self.min_hold_minutes = sig_config.get('min_hold_minutes', 2)
        self.max_signals_per_hour = sig_config.get('max_signals_per_hour', 12)
        self.cooldown_minutes = sig_config.get('cooldown_minutes', 5)
        
        # Trend filter
        trend_config = sig_config.get('trend_filter', {})
        self.trend_filter_enabled = trend_config.get('enabled', True)
        self.trend_ma_period = trend_config.get('ma_period', 200)
        self.trend_slope_threshold = trend_config.get('slope_threshold', 0.0001)
        
        # Volatility filter
        vol_config = sig_config.get('volatility_filter', {})
        self.volatility_filter_enabled = vol_config.get('enabled', True)
        self.atr_threshold = vol_config.get('atr_threshold', 0.001)
        
        # Volume filter
        vol_config = sig_config.get('volume_filter', {})
        self.volume_filter_enabled = vol_config.get('enabled', True)
        self.volume_ma_period = vol_config.get('volume_ma_period', 20)
        self.volume_threshold = vol_config.get('volume_threshold', 1.2)
        
        # Session filter
        sess_config = sig_config.get('session_filter', {})
        self.session_filter_enabled = sess_config.get('enabled', True)
        self.allowed_sessions = sess_config.get('allowed_sessions', ['london', 'new_york', 'overlap'])
        
        # Rate limiting
        self.current_hour = -1
        self.signals_this_hour = 0
        self.last_signal_time = None
        self.last_signal_type = None
    
    def generate(
        self,
        features: pd.DataFrame,
        ohlcv: pd.DataFrame,
    ) -> List[Signal]:
        """Generate signals from features."""
        signals = []
        
        for i in range(len(features)):
            timestamp = features.index[i]
            feature_row = features.iloc[i:i+1]
            ohlcv_row = ohlcv.loc[timestamp] if timestamp in ohlcv.index else ohlcv.iloc[i]
            
            # Check filters
            if not self._passes_filters(timestamp, feature_row, ohlcv_row):
                signals.append(Signal(
                    timestamp=timestamp,
                    signal_type=SignalType.WAIT,
                    confidence=0.0,
                    price=ohlcv_row['close'],
                ))
                continue
            
            # Get prediction
            probas = self._get_probabilities(feature_row)
            if probas is None:
                signals.append(Signal(
                    timestamp=timestamp,
                    signal_type=SignalType.WAIT,
                    confidence=0.0,
                    price=ohlcv_row['close'],
                ))
                continue
            
            # Convert to signal
            signal_type, confidence = self._proba_to_signal(probas)
            
            # Apply confidence threshold
            if confidence < self.confidence_threshold:
                signal_type = SignalType.WAIT
                confidence = 0.0
            
            # Apply smoothing
            if len(signals) >= self.signal_smoothing:
                signal_type = self._apply_smoothing(signal_type, signals)
            
            # Check rate limits
            if signal_type != SignalType.WAIT:
                if not self._check_rate_limits(timestamp):
                    signal_type = SignalType.WAIT
                    confidence = 0.0
                else:
                    self._update_rate_limits(timestamp)
            
            signals.append(Signal(
                timestamp=timestamp,
                signal_type=signal_type,
                confidence=confidence,
                price=ohlcv_row['close'],
                features=feature_row.iloc[0].to_dict(),
            ))
        
        return signals
    
    def _passes_filters(
        self,
        timestamp: pd.Timestamp,
        features: pd.DataFrame,
        ohlcv_row: pd.Series,
    ) -> bool:
        """Check all signal filters."""
        # Trend filter
        if self.trend_filter_enabled and 'close' in ohlcv_row:
            ma = ohlcv_row['close'].rolling(self.trend_ma_period).mean()
            if not pd.isna(ma):
                slope = (ohlcv_row['close'] - ma) / ma
                if abs(slope) < self.trend_slope_threshold:
                    return False
        
        # Volatility filter
        if self.volatility_filter_enabled:
            atr = features.get('atr_14', pd.Series([self.atr_threshold])).iloc[0]
            if atr < self.atr_threshold:
                return False
        
        # Volume filter
        if self.volume_filter_enabled:
            vol_ratio = features.get('volume_ratio', pd.Series([1.0])).iloc[0]
            if vol_ratio < self.volume_threshold:
                return False
        
        # Session filter
        if self.session_filter_enabled:
            if not self._is_allowed_session(timestamp):
                return False
        
        return True
    
    def _is_allowed_session(self, timestamp: pd.Timestamp) -> bool:
        """Check if timestamp is in allowed trading session."""
        hour = timestamp.hour
        minute = timestamp.minute
        time_mins = hour * 60 + minute
        
        sessions = {
            'london': (8 * 60, 17 * 60),
            'new_york': (13 * 60, 22 * 60),
            'overlap': (13 * 60, 17 * 60),
            'asian': (0, 8 * 60),
        }
        
        for session in self.allowed_sessions:
            if session in sessions:
                start, end = sessions[session]
                if start <= time_mins < end:
                    return True
        
        return False
    
    def _get_probabilities(self, features: pd.DataFrame) -> Optional[np.ndarray]:
        """Get prediction probabilities."""
        try:
            if self.feature_names:
                features = features[self.feature_names]
            
            if self.scaler:
                features_scaled = self.scaler.transform(features)
            else:
                features_scaled = features.values
            
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(features_scaled)
            return None
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None
    
    def _proba_to_signal(self, probas: np.ndarray) -> tuple:
        """Convert probabilities to signal type and confidence."""
        pred_class = np.argmax(probas[0])
        confidence = probas[0][pred_class]
        
        if pred_class == 0:  # SELL
            return SignalType.SELL, confidence
        elif pred_class == 2:  # BUY
            return SignalType.BUY, confidence
        else:  # WAIT
            return SignalType.WAIT, 0.0
    
    def _apply_smoothing(
        self,
        signal_type: SignalType,
        recent_signals: List[Signal],
    ) -> SignalType:
        """Apply signal smoothing."""
        if signal_type == SignalType.WAIT:
            return signal_type
        
        window = recent_signals[-self.signal_smoothing:]
        non_wait = [s for s in window if s.signal_type != SignalType.WAIT]
        
        if len(non_wait) >= self.signal_smoothing // 2 + 1:
            return non_wait[-1].signal_type
        
        return SignalType.WAIT
    
    def _check_rate_limits(self, timestamp: pd.Timestamp) -> bool:
        """Check hourly rate limit and cooldown."""
        hour = timestamp.hour
        
        if self.current_hour != hour:
            self.current_hour = hour
            self.signals_this_hour = 0
        
        if self.signals_this_hour >= self.max_signals_per_hour:
            return False
        
        if self.last_signal_time is not None:
            minutes_since = (timestamp - self.last_signal_time).total_seconds() / 60
            if minutes_since < self.cooldown_minutes:
                return False
        
        return True
    
    def _update_rate_limits(self, timestamp: pd.Timestamp) -> None:
        """Update rate limit counters."""
        self.signals_this_hour += 1
        self.last_signal_time = timestamp
    
    def get_signal_series(self, signals: List[Signal]) -> pd.Series:
        """Convert signals to pandas series."""
        signal_values = []
        timestamps = []
        
        for s in signals:
            if s.signal_type == SignalType.BUY:
                signal_values.append(1)
            elif s.signal_type == SignalType.SELL:
                signal_values.append(-1)
            else:
                signal_values.append(0)
            timestamps.append(s.timestamp)
        
        return pd.Series(signal_values, index=timestamps, name='signal')
    
    def get_confidence_series(self, signals: List[Signal]) -> pd.Series:
        """Get confidence series."""
        confidences = [s.confidence for s in signals]
        timestamps = [s.timestamp for s in signals]
        return pd.Series(confidences, index=timestamps, name='confidence')


class EnsembleSignalGenerator(SignalGenerator):
    """Signal generator for ensemble models."""
    
    def __init__(
        self,
        ensemble_model: Any,
        config_path: Optional[str] = None,
        scalers: Dict[str, Any] = None,
        feature_names: List[str] = None,
    ):
        from models.ensemble import EnsembleModel
        self.ensemble_model = ensemble_model
        self.scalers = scalers or {}
        
        super().__init__(
            model=ensemble_model,
            config_path=config_path,
            scaler=None,
            feature_names=feature_names,
        )
    
    def _proba_to_signal(self, probas: np.ndarray) -> tuple:
        """Convert ensemble probabilities to signal."""
        pred_class = np.argmax(probas[0])
        confidence = probas[0][pred_class]
        
        if pred_class == 0:
            return SignalType.SELL, confidence
        elif pred_class == 2:
            return SignalType.BUY, confidence
        else:
            return SignalType.WAIT, 0.0
    
    def get_individual_predictions(self, features: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Get predictions from individual models."""
        return self.ensemble_model.get_model_predictions(features)
    
    def get_individual_probabilities(self, features: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Get probabilities from individual models."""
        return self.ensemble_model.get_model_probabilities(features)


def create_signal_generator(
    model_path: str,
    config_path: Optional[str] = None,
) -> SignalGenerator:
    """Create signal generator from saved model."""
    from ..models.persistence import ModelPersistence
    
    persistence = ModelPersistence()
    loaded = persistence.load_latest(model_path)
    
    return SignalGenerator(
        model=loaded['model'],
        config_path=config_path,
        scaler=loaded['scaler'],
        feature_names=loaded['feature_names'],
    )


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    from sklearn.linear_model import LogisticRegression
    
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn(n_samples, 20)
    y = np.random.choice([-1, 0, 1], n_samples, p=[0.2, 0.6, 0.2])
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    
    features = pd.DataFrame(X, columns=[f'f_{i}' for i in range(20)])
    features.index = pd.date_range('2024-01-01', periods=n_samples, freq='1min')
    
    ohlcv = pd.DataFrame({
        'close': 2000 + np.cumsum(np.random.randn(n_samples) * 0.1),
    }, index=features.index)
    ohlcv['open'] = ohlcv['close'].shift(1).fillna(ohlcv['close'].iloc[0])
    ohlcv['high'] = ohlcv[['open', 'close']].max(axis=1) + np.abs(np.random.randn(n_samples) * 0.05)
    ohlcv['low'] = ohlcv[['open', 'close']].min(axis=1) - np.abs(np.random.randn(n_samples) * 0.05)
    ohlcv['volume'] = np.random.randint(100, 1000, n_samples)
    
    generator = SignalGenerator(model)
    signals = generator.generate(features, ohlcv)
    
    signal_series = generator.get_signal_series(signals)
    print(f"Signals: {signal_series.value_counts().to_dict()}")
    print(f"Total trades: {len([s for s in signals if s.signal_type != SignalType.WAIT])}")
