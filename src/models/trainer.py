"""Model training for XAUUSD Scalping System."""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import joblib

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str
    model_class: str
    params: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    enabled: bool = True


class ModelTrainer:
    """Train and evaluate multiple models."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize trainer."""
        from ..config import get_config
        self.config = get_config(config_path)
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_names: List[str] = []
        self.best_model_name: Optional[str] = None
        self._results: Dict[str, Dict[str, float]] = {}
        
        # Load model configurations
        model_config = self.config.get_section('models')
        self.model_configs = self._parse_model_configs(model_config)
    
    def _parse_model_configs(self, config: Dict[str, Any]) -> List[ModelConfig]:
        """Parse model configurations."""
        configs = []
        
        # Default configurations
        defaults = {
            'logistic_regression': {
                'model_class': 'LogisticRegression',
                'params': {'max_iter': 1000, 'C': 1.0, 'class_weight': 'balanced'},
                'weight': 1.0,
                'enabled': True,
            },
            'random_forest': {
                'model_class': 'RandomForestClassifier',
                'params': {'n_estimators': 100, 'max_depth': 10, 'class_weight': 'balanced'},
                'weight': 1.0,
                'enabled': True,
            },
            'gradient_boosting': {
                'model_class': 'GradientBoostingClassifier',
                'params': {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1},
                'weight': 1.0,
                'enabled': True,
            },
        }
        
        # Check for XGBoost
        try:
            import xgboost as xgb
            defaults['xgboost'] = {
                'model_class': 'XGBClassifier',
                'params': {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1},
                'weight': 1.5,
                'enabled': True,
            }
        except ImportError:
            pass
        
        # Check for LightGBM
        try:
            import lightgbm as lgb
            defaults['lightgbm'] = {
                'model_class': 'LGBMClassifier',
                'params': {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1},
                'weight': 1.5,
                'enabled': True,
            }
        except ImportError:
            pass
        
        # Check for CatBoost
        try:
            from catboost import CatBoostClassifier
            defaults['catboost'] = {
                'model_class': 'CatBoostClassifier',
                'params': {'iterations': 100, 'depth': 5, 'learning_rate': 0.1},
                'weight': 1.5,
                'enabled': True,
            }
        except ImportError:
            pass
        
        # Override with config if provided
        for name, cfg in config.items():
            if isinstance(cfg, dict):
                defaults[name] = cfg
        
        for name, cfg in defaults.items():
            configs.append(ModelConfig(
                name=name,
                model_class=cfg.get('model_class', name),
                params=cfg.get('params', {}),
                weight=cfg.get('weight', 1.0),
                enabled=cfg.get('enabled', True),
            ))
        
        return configs
    
    def _create_model(self, model_config: ModelConfig):
        """Create a model instance from configuration."""
        model_class = model_config.model_class
        
        if model_class == 'LogisticRegression':
            return LogisticRegression(**model_config.params)
        elif model_class == 'RandomForestClassifier':
            return RandomForestClassifier(**model_config.params)
        elif model_class == 'GradientBoostingClassifier':
            return GradientBoostingClassifier(**model_config.params)
        elif model_class == 'XGBClassifier':
            import xgboost as xgb
            return xgb.XGBClassifier(**model_config.params)
        elif model_class == 'LGBMClassifier':
            import lightgbm as lgb
            return lgb.LGBMClassifier(**model_config.params)
        elif model_class == 'CatBoostClassifier':
            from catboost import CatBoostClassifier
            return CatBoostClassifier(**model_config.params)
        else:
            raise ValueError(f"Unknown model class: {model_class}")
    
    def _evaluate_model(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int = 5,
    ) -> Dict[str, float]:
        """Evaluate model using time series cross-validation."""
        from sklearn.model_selection import cross_val_score
        
        try:
            scores = cross_val_score(
                model, X, y,
                cv=TimeSeriesSplit(n_splits=cv_folds),
                scoring='accuracy',
            )
            
            return {
                'accuracy_mean': float(np.mean(scores)),
                'accuracy_std': float(np.std(scores)),
                'accuracy_min': float(np.min(scores)),
                'accuracy_max': float(np.max(scores)),
            }
        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}")
            return {
                'accuracy_mean': 0.5,
                'accuracy_std': 0.0,
                'accuracy_min': 0.5,
                'accuracy_max': 0.5,
            }
    
    def _get_feature_importance(self, model: Any, feature_names: List[str]) -> pd.DataFrame:
        """Get feature importance from model."""
        importance = None
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_).mean(axis=0)
        
        if importance is not None:
            return pd.DataFrame({
                'feature': feature_names,
                'importance': importance,
            }).sort_values('importance', ascending=False)
        
        return pd.DataFrame({'feature': feature_names, 'importance': 0.0})
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str,
        test_size: float = 0.2,
    ) -> Tuple[Any, Dict[str, float], pd.DataFrame]:
        """Train a single model."""
        # Scale features
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index,
        )
        
        # Split data
        split_idx = int(len(X_scaled) * (1 - test_size))
        X_train, X_test = X_scaled.iloc[:split_idx], X_scaled.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Find config
        model_config = None
        for cfg in self.model_configs:
            if cfg.name == model_name:
                model_config = cfg
                break
        
        if model_config is None:
            model_config = ModelConfig(name=model_name, model_class=model_name)
        
        # Create and train model
        model = self._create_model(model_config)
        model.fit(X_train, y_train)
        
        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        metrics = {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
        }
        
        # Cross-validation
        cv_metrics = self._evaluate_model(model, X_scaled, y, cv_folds=5)
        metrics.update(cv_metrics)
        
        # Feature importance
        importance = self._get_feature_importance(model, list(X.columns))
        
        return model, metrics, importance
    
    def train_all(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
    ) -> Dict[str, Dict[str, float]]:
        """Train all enabled models."""
        results = {}
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Filter enabled models
        enabled_models = [cfg for cfg in self.model_configs if cfg.enabled]
        
        if not enabled_models:
            logger.warning("No enabled models to train")
            return results
        
        # Train each model
        for model_config in enabled_models:
            logger.info(f"Training {model_config.name}...")
            try:
                model, metrics, importance = self.train(
                    X, y,
                    model_name=model_config.name,
                    test_size=test_size,
                )
                
                self.models[model_config.name] = model
                self._results[model_config.name] = metrics
                
                # Store scaler
                scaler = StandardScaler()
                scaler.fit(X)
                self.scalers[model_config.name] = scaler
                
                results[model_config.name] = metrics
                logger.info(f"{model_config.name}: test_acc={metrics.get('test_accuracy', 0):.4f}")
                
            except Exception as e:
                logger.error(f"Failed to train {model_config.name}: {e}")
                results[model_config.name] = {'error': str(e)}
        
        # Find best model
        if self._results:
            best_name = max(
                self._results.keys(),
                key=lambda k: self._results[k].get('test_accuracy', 0),
            )
            self.best_model_name = best_name
            logger.info(f"Best model: {best_name}")
        
        return results
    
    def get_feature_importance(self, model_name: Optional[str] = None) -> pd.DataFrame:
        """Get feature importance for a model."""
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.models:
            return pd.DataFrame({'feature': [], 'importance': []})
        
        model = self.models[model_name]
        return self._get_feature_importance(model, self.feature_names)
    
    def predict(self, X: pd.DataFrame, model_name: Optional[str] = None) -> np.ndarray:
        """Make predictions."""
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.models:
            raise ValueError(f"Model not found: {model_name}")
        
        model = self.models[model_name]
        scaler = self.scalers.get(model_name)
        
        if scaler:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X
        
        return model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame, model_name: Optional[str] = None) -> np.ndarray:
        """Get prediction probabilities."""
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.models:
            raise ValueError(f"Model not found: {model_name}")
        
        model = self.models[model_name]
        scaler = self.scalers.get(model_name)
        
        if scaler:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X
        
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X_scaled)
        return None
    
    def save_model(
        self,
        model_name: Optional[str] = None,
        output_dir: str = "models",
    ) -> str:
        """Save a trained model."""
        from .persistence import ModelPersistence
        
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.models:
            raise ValueError(f"Model not found: {model_name}")
        
        persistence = ModelPersistence(output_dir)
        
        path = persistence.save(
            model=self.models[model_name],
            model_name=model_name,
            scaler=self.scalers.get(model_name),
            feature_names=self.feature_names,
            metrics=self._results.get(model_name, {}),
            feature_importance=self.get_feature_importance(model_name),
            config=self.config.get_section('models'),
        )
        
        return path


if __name__ == "__main__":
    # Test trainer
    import logging
    logging.basicConfig(level=logging.INFO)
    
    np.random.seed(42)
    n_samples = 1000
    X = pd.DataFrame(
        np.random.randn(n_samples, 10),
        columns=[f"feature_{i}" for i in range(10)],
    )
    y = pd.Series(np.random.choice([0, 1, 2], n_samples))
    
    trainer = ModelTrainer()
    results = trainer.train_all(X, y, test_size=0.2)
    
    print(f"Trained models: {list(trainer.models.keys())}")
    print(f"Best model: {trainer.best_model_name}")
    print(f"Results: {results}")
