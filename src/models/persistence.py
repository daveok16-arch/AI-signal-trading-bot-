"""Model persistence for XAUUSD Scalping System."""
import os
import json
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import shutil

logger = logging.getLogger(__name__)


class ModelPersistence:
    """Handle model save/load operations."""
    
    def __init__(self, model_dir: Optional[str] = None):
        """Initialize persistence with model directory."""
        self.model_dir = Path(model_dir) if model_dir else Path("models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """Ensure model subdirectories exist."""
        (self.model_dir / "versions").mkdir(parents=True, exist_ok=True)
        (self.model_dir / "metadata").mkdir(parents=True, exist_ok=True)
    
    def _get_model_path(self, model_name: str, version: Optional[str] = None) -> Path:
        """Get path to model file."""
        if version:
            return self.model_dir / "versions" / f"{model_name}_{version}.joblib"
        return self.model_dir / "versions" / f"{model_name}_latest.joblib"
    
    def _get_metadata_path(self, model_name: str, version: Optional[str] = None) -> Path:
        """Get path to metadata file."""
        if version:
            return self.model_dir / "metadata" / f"{model_name}_{version}.json"
        return self.model_dir / "metadata" / f"{model_name}_latest.json"
    
    def save(
        self,
        model: Any,
        model_name: str,
        scaler: Optional[Any] = None,
        feature_names: Optional[List[str]] = None,
        metrics: Optional[Dict[str, float]] = None,
        feature_importance: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        version: Optional[str] = None,
    ) -> str:
        """Save model and associated artifacts."""
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_path = self._get_model_path(model_name, version)
        metadata_path = self._get_metadata_path(model_name, version)
        
        # Save model
        joblib.dump(model, model_path)
        logger.info(f"Saved model to {model_path}")
        
        # Save scaler alongside model if present
        if scaler is not None:
            scaler_path = model_path.with_suffix('.scaler.joblib')
            joblib.dump(scaler, scaler_path)
        
        # Create metadata
        metadata = {
            "model_name": model_name,
            "version": version,
            "saved_at": datetime.now().isoformat(),
            "metrics": metrics or {},
            "feature_names": feature_names or [],
            "feature_importance": (
                feature_importance.to_dict() if hasattr(feature_importance, 'to_dict') 
                else (feature_importance if feature_importance is not None else {})
            ),
            "config": config or {},
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update latest symlink/copy
        latest_model_path = self._get_model_path(model_name)
        latest_metadata_path = self._get_metadata_path(model_name)
        
        shutil.copy2(model_path, latest_model_path)
        shutil.copy2(metadata_path, latest_metadata_path)
        
        if scaler is not None:
            latest_scaler_path = latest_model_path.with_suffix('.scaler.joblib')
            shutil.copy2(scaler_path, latest_scaler_path)
        
        logger.info(f"Saved model version: {version}")
        return str(model_path)
    
    def load(self, model_name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """Load model and associated artifacts."""
        model_path = self._get_model_path(model_name, version)
        metadata_path = self._get_metadata_path(model_name, version)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_name} (version: {version})")
        
        # Load model
        model = joblib.load(model_path)
        
        # Load metadata
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        # Load scaler if exists
        scaler = None
        scaler_path = model_path.with_suffix('.scaler.joblib')
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
        
        return {
            "model": model,
            "scaler": scaler,
            "feature_names": metadata.get("feature_names", []),
            "metrics": metadata.get("metrics", {}),
            "feature_importance": metadata.get("feature_importance", {}),
            "version": metadata.get("version", version),
            "saved_at": metadata.get("saved_at"),
        }
    
    def load_latest(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Load latest version of a model."""
        try:
            return self.load(model_name, version=None)
        except FileNotFoundError:
            logger.warning(f"No model found for: {model_name}")
            return None
    
    def list_models(self) -> List[str]:
        """List available model names."""
        models = set()
        versions_dir = self.model_dir / "versions"
        
        if versions_dir.exists():
            for f in versions_dir.glob("*_latest.joblib"):
                model_name = f.stem.replace("_latest", "")
                models.add(model_name)
        
        return sorted(list(models))
    
    def list_versions(self, model_name: str) -> List[str]:
        """List available versions for a model."""
        versions = []
        versions_dir = self.model_dir / "versions"
        
        if versions_dir.exists():
            pattern = f"{model_name}_*.joblib"
            for f in versions_dir.glob(pattern):
                version = f.stem.replace(f"{model_name}_", "")
                if version != "latest":
                    versions.append(version)
        
        return sorted(versions, reverse=True)
    
    def delete(self, model_name: str, version: str) -> bool:
        """Delete a model version."""
        model_path = self._get_model_path(model_name, version)
        metadata_path = self._get_metadata_path(model_name, version)
        scaler_path = model_path.with_suffix('.scaler.joblib')
        
        deleted = False
        
        if model_path.exists():
            model_path.unlink()
            deleted = True
        
        if metadata_path.exists():
            metadata_path.unlink()
        
        if scaler_path.exists():
            scaler_path.unlink()
        
        if deleted:
            logger.info(f"Deleted model: {model_name} version: {version}")
        
        return deleted
    
    def get_model_info(self, model_name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """Get model metadata without loading the model."""
        metadata_path = self._get_metadata_path(model_name, version)
        
        if not metadata_path.exists():
            return {}
        
        with open(metadata_path, 'r') as f:
            return json.load(f)


if __name__ == "__main__":
    # Test persistence
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    
    persistence = ModelPersistence("test_models")
    
    # Create dummy model
    X = np.random.randn(100, 5)
    y = np.random.choice([0, 1, 2], 100)
    model = LogisticRegression()
    model.fit(X, y)
    
    # Save
    path = persistence.save(
        model=model,
        model_name="test_model",
        feature_names=[f"f_{i}" for i in range(5)],
        metrics={"accuracy": 0.85},
    )
    
    # Load
    loaded = persistence.load_latest("test_model")
    print(f"Loaded model: {loaded['version']}")
    print(f"Feature names: {loaded['feature_names']}")
    
    # List
    print(f"Models: {persistence.list_models()}")
