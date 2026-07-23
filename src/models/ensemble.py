"""Ensemble models for XAUUSD Scalping System."""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class EnsembleConfig:
    """Configuration for ensemble model."""
    voting_method: str = "soft"  # "soft", "hard", or "weighted"
    weights: Optional[List[float]] = None
    require_consensus: bool = True
    consensus_threshold: float = 0.6


class EnsembleModel:
    """Ensemble of multiple models."""
    
    def __init__(
        self,
        models: Dict[str, Any],
        scalers: Optional[Dict[str, Any]] = None,
        feature_names: Optional[List[str]] = None,
        config: Optional[EnsembleConfig] = None,
    ):
        """Initialize ensemble."""
        self.models = models
        self.scalers = scalers or {}
        self.feature_names = feature_names or []
        self.config = config or EnsembleConfig()
        self._validate_models()
    
    def _validate_models(self) -> None:
        """Validate that all models have required methods."""
        for name, model in self.models.items():
            if not hasattr(model, 'predict'):
                raise ValueError(f"Model {name} does not have predict method")
            if not hasattr(model, 'predict_proba'):
                logger.warning(f"Model {name} does not have predict_proba - using hard voting")
                self.config.voting_method = "hard"
    
    def _get_model_weights(self) -> List[float]:
        """Get model weights for weighted voting."""
        if self.config.weights and len(self.config.weights) == len(self.models):
            return self.config.weights
        
        # Equal weights
        n_models = len(self.models)
        return [1.0 / n_models] * n_models
    
    def get_model_predictions(self, features: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Get predictions from each model."""
        predictions = {}
        
        for name, model in self.models.items():
            try:
                scaler = self.scalers.get(name)
                if scaler:
                    X = scaler.transform(features)
                else:
                    X = features
                
                if hasattr(model, 'predict'):
                    predictions[name] = model.predict(X)
            except Exception as e:
                logger.error(f"Failed to get prediction from {name}: {e}")
        
        return predictions
    
    def get_model_probabilities(self, features: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Get probability predictions from each model."""
        probabilities = {}
        
        for name, model in self.models.items():
            try:
                scaler = self.scalers.get(name)
                if scaler:
                    X = scaler.transform(features)
                else:
                    X = features
                
                if hasattr(model, 'predict_proba'):
                    probabilities[name] = model.predict_proba(X)
                elif hasattr(model, 'predict'):
                    # Convert hard predictions to probabilities
                    pred = model.predict(X)
                    proba = np.zeros((len(pred), 3))
                    for i, p in enumerate(pred):
                        proba[i, int(p)] = 1.0
                    probabilities[name] = proba
            except Exception as e:
                logger.error(f"Failed to get probabilities from {name}: {e}")
        
        return probabilities
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions."""
        predictions = self.get_model_predictions(features)
        
        if not predictions:
            return np.zeros(len(features), dtype=int)
        
        if self.config.voting_method == "hard":
            return self._hard_voting(predictions)
        else:
            probabilities = self.get_model_probabilities(features)
            return self._soft_voting(probabilities)
    
    def _hard_voting(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Hard voting - majority vote."""
        model_names = list(predictions.keys())
        weights = self._get_model_weights()
        
        n_samples = len(list(predictions.values())[0])
        n_classes = 3  # SELL, WAIT, BUY
        
        votes = np.zeros((n_samples, n_classes))
        
        for name, pred in predictions.items():
            weight = weights[model_names.index(name)]
            for i, p in enumerate(pred):
                votes[i, int(p)] += weight
        
        return np.argmax(votes, axis=1)
    
    def _soft_voting(self, probabilities: Dict[str, np.ndarray]) -> np.ndarray:
        """Soft voting - average probabilities."""
        model_names = list(probabilities.keys())
        weights = self._get_model_weights()
        
        n_samples = len(list(probabilities.values())[0])
        n_classes = 3
        
        avg_proba = np.zeros((n_samples, n_classes))
        
        for name, proba in probabilities.items():
            weight = weights[model_names.index(name)]
            avg_proba += proba * weight
        
        if self.config.require_consensus:
            max_proba = np.max(avg_proba, axis=1)
            predictions = np.argmax(avg_proba, axis=1)
            
            # If no model agrees, predict WAIT (class 1)
            mask = max_proba < self.config.consensus_threshold
            predictions[mask] = 1
            
            return predictions
        
        return np.argmax(avg_proba, axis=1)
    
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Get ensemble probability predictions."""
        probabilities = self.get_model_probabilities(features)
        
        if not probabilities:
            return np.zeros((len(features), 3))
        
        model_names = list(probabilities.keys())
        weights = self._get_model_weights()
        
        n_samples = len(list(probabilities.values())[0])
        avg_proba = np.zeros((n_samples, 3))
        
        for name, proba in probabilities.items():
            weight = weights[model_names.index(name)]
            avg_proba += proba * weight
        
        return avg_proba
    
    def get_agreement_scores(self, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate how much models agree on each prediction."""
        predictions = self.get_model_predictions(features)
        
        if not predictions:
            return pd.DataFrame()
        
        n_samples = len(list(predictions.values())[0])
        n_models = len(predictions)
        
        # Count agreements
        agreement = np.zeros(n_samples)
        model_names = list(predictions.keys())
        
        for i in range(n_samples):
            votes = [predictions[name][i] for name in model_names]
            most_common = max(set(votes), key=votes.count)
            agreement[i] = votes.count(most_common) / n_models
        
        return pd.DataFrame({
            'sample_index': range(n_samples),
            'agreement': agreement,
        })
    
    def get_model_contributions(self, features: pd.DataFrame) -> Dict[str, float]:
        """Get contribution scores for each model based on recent performance."""
        # Simple contribution based on prediction variance
        predictions = self.get_model_predictions(features)
        
        if not predictions:
            return {}
        
        model_names = list(predictions.keys())
        n_models = len(model_names)
        
        # Calculate how often each model is in the majority
        contributions = {name: 0.0 for name in model_names}
        
        n_samples = len(list(predictions.values())[0])
        for i in range(n_samples):
            votes = [predictions[name][i] for name in model_names]
            majority = max(set(votes), key=votes.count)
            
            # Models in majority get contribution
            for name in model_names:
                if predictions[name][i] == majority:
                    contributions[name] += 1
        
        # Normalize
        total = sum(contributions.values())
        if total > 0:
            contributions = {k: v / total for k, v in contributions.items()}
        
        return contributions


class StackingEnsemble(EnsembleModel):
    """Stacking ensemble with meta-learner."""
    
    def __init__(
        self,
        base_models: Dict[str, Any],
        meta_model: Any,
        scalers: Optional[Dict[str, Any]] = None,
        feature_names: Optional[List[str]] = None,
        config: Optional[EnsembleConfig] = None,
    ):
        """Initialize stacking ensemble."""
        super().__init__(base_models, scalers, feature_names, config)
        self.meta_model = meta_model
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
    ) -> None:
        """Fit the stacking ensemble."""
        from sklearn.model_selection import train_test_split
        
        # Split for meta-features
        X_train, X_meta, y_train, y_meta = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        
        # Get base model predictions on meta set
        meta_features = self._get_meta_features(X_meta)
        
        # Fit meta-model
        self.meta_model.fit(meta_features, y_meta)
    
    def _get_meta_features(self, X: pd.DataFrame) -> np.ndarray:
        """Generate meta-features from base model predictions."""
        probabilities = self.get_model_probabilities(X)
        
        if not probabilities:
            return np.zeros((len(X), 3 * len(self.models)))
        
        # Concatenate all probabilities
        model_names = sorted(probabilities.keys())
        meta_features = np.hstack([probabilities[name] for name in model_names])
        
        return meta_features
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using stacking."""
        meta_features = self._get_meta_features(X)
        return self.meta_model.predict(meta_features)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get probabilities using stacking."""
        meta_features = self._get_meta_features(X)
        
        if hasattr(self.meta_model, 'predict_proba'):
            return self.meta_model.predict_proba(meta_features)
        
        pred = self.meta_model.predict(meta_features)
        proba = np.zeros((len(pred), 3))
        for i, p in enumerate(pred):
            proba[i, int(p)] = 1.0
        return proba


if __name__ == "__main__":
    # Test ensemble
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    
    np.random.seed(42)
    n_samples = 100
    
    X = pd.DataFrame(
        np.random.randn(n_samples, 5),
        columns=[f"f_{i}" for i in range(5)],
    )
    y = np.random.choice([0, 1, 2], n_samples)
    
    # Create simple models
    models = {
        'lr': LogisticRegression().fit(X, y),
        'rf': RandomForestClassifier().fit(X, y),
    }
    
    # Create ensemble
    ensemble = EnsembleModel(models)
    
    # Predict
    predictions = ensemble.predict(X)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Prediction distribution: {np.bincount(predictions)}")
