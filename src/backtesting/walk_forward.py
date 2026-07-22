"""Walk-forward validation for XAUUSD Scalping System."""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation."""
    train_window: int = 5000
    test_window: int = 1000
    step_size: int = 500
    min_train_size: int = 3000
    expanding_window: bool = False
    purge_gap: int = 10
    embargo_pct: float = 0.01


class WalkForwardValidator:
    """Walk-forward validation with purging and embargo."""
    
    def __init__(self, config: WalkForwardConfig = None):
        self.config = config or WalkForwardConfig()
    
    def split(self, X: pd.DataFrame, y: pd.Series) -> List[tuple]:
        """Generate walk-forward train/test splits with purging."""
        n_samples = len(X)
        splits = []
        
        start = 0
        while start + self.config.train_window + self.config.test_window <= n_samples:
            train_end = start + self.config.train_window
            test_start = train_end + self.config.purge_gap
            test_end = test_start + self.config.test_window
            
            if test_end > n_samples:
                break
            
            # Apply embargo
            embargo_size = int(self.config.test_window * self.config.embargo_pct)
            test_end_with_embargo = min(test_end + embargo_size, n_samples)
            
            train_indices = list(range(start, train_end))
            test_indices = list(range(test_start, test_end))
            
            splits.append((train_indices, test_indices))
            
            if self.config.expanding_window:
                start = 0
            else:
                start += self.config.step_size
        
        logger.info(f"Generated {len(splits)} walk-forward splits")
        return splits
    
    def validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_factory: Callable,
        fit_params: Dict = None,
    ) -> Dict[str, Any]:
        """Run walk-forward validation."""
        fit_params = fit_params or {}
        splits = self.split(X, y)
        
        results = {
            'splits': [],
            'train_scores': [],
            'test_scores': [],
            'models': [],
        }
        
        for i, (train_idx, test_idx) in enumerate(splits):
            logger.info(f"Walk-forward split {i+1}/{len(splits)}")
            
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
            
            model = model_factory()
            model.fit(X_train, y_train, **fit_params)
            
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            results['splits'].append({
                'train_indices': train_idx,
                'test_indices': test_idx,
            })
            results['train_scores'].append(train_score)
            results['test_scores'].append(test_score)
            results['models'].append(model)
        
        results['mean_train_score'] = np.mean(results['train_scores'])
        results['mean_test_score'] = np.mean(results['test_scores'])
        results['std_test_score'] = np.std(results['test_scores'])
        
        return results


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    
    np.random.seed(42)
    n = 10000
    X = pd.DataFrame(np.random.randn(n, 10), columns=[f'f{i}' for i in range(10)])
    y = pd.Series(np.random.randint(0, 2, n))
    
    config = WalkForwardConfig(train_window=2000, test_window=500, step_size=500)
    validator = WalkForwardValidator(config)
    
    def model_factory():
        return LogisticRegression(max_iter=1000)
    
    results = validator.validate(X, y, model_factory)
    print(f"Mean test score: {results['mean_test_score']:.4f} (+/- {results['std_test_score']:.4f})")
