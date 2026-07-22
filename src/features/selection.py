"""Feature selection for XAUUSD Scalping System."""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.feature_selection import (
    SelectKBest, mutual_info_classif, mutual_info_regression,
    f_classif, f_regression, VarianceThreshold, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class FeatureSelector:
    """Feature selection pipeline."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
        self.method = self.config.get('method', 'mutual_info')
        self.k_best = self.config.get('k_best', 50)
        self.correlation_threshold = self.config.get('correlation_threshold', 0.95)
        self.variance_threshold = self.config.get('variance_threshold', 0.001)
        
        self.selector = None
        self.selected_features: List[str] = []
        self.feature_scores: Dict[str, float] = {}
    
    def fit(self, X: pd.DataFrame, y: pd.Series, task_type: str = 'classification') -> 'FeatureSelector':
        """Fit feature selector."""
        if not self.enabled:
            self.selected_features = X.columns.tolist()
            return self
        
        X_clean = X.copy()
        
        # Remove constant/near-constant features
        vt = VarianceThreshold(threshold=self.variance_threshold)
        X_var = vt.fit_transform(X_clean)
        var_features = X_clean.columns[vt.get_support()].tolist()
        X_clean = X_clean[var_features]
        
        logger.info(f"After variance threshold: {len(X_clean.columns)} features")
        
        # Remove highly correlated features
        X_clean = self._remove_correlated(X_clean)
        
        logger.info(f"After correlation removal: {len(X_clean.columns)} features")
        
        # Select K best features
        if self.method == 'mutual_info':
            if task_type == 'classification':
                score_func = mutual_info_classif
            else:
                score_func = mutual_info_regression
        elif self.method == 'f_statistic':
            if task_type == 'classification':
                score_func = f_classif
            else:
                score_func = f_regression
        else:
            score_func = mutual_info_classif
        
        k = min(self.k_best, len(X_clean.columns))
        self.selector = SelectKBest(score_func=score_func, k=k)
        self.selector.fit(X_clean, y)
        
        self.selected_features = X_clean.columns[self.selector.get_support()].tolist()
        
        # Store scores
        scores = self.selector.scores_
        self.feature_scores = dict(zip(X_clean.columns, scores))
        
        # Sort by score
        self.feature_scores = dict(
            sorted(self.feature_scores.items(), key=lambda x: x[1], reverse=True)
        )
        
        logger.info(f"Selected {len(self.selected_features)} features")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform feature matrix."""
        if not self.enabled:
            return X
        
        if not self.selected_features:
            logger.warning("Selector not fitted, returning all features")
            return X
        
        # Ensure all selected features exist
        available = [f for f in self.selected_features if f in X.columns]
        missing = set(self.selected_features) - set(available)
        if missing:
            logger.warning(f"Missing features: {missing}")
        
        return X[available]
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series, task_type: str = 'classification') -> pd.DataFrame:
        """Fit and transform."""
        self.fit(X, y, task_type)
        return self.transform(X)
    
    def _remove_correlated(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove highly correlated features."""
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = [col for col in upper.columns if any(upper[col] > self.correlation_threshold)]
        
        if to_drop:
            logger.info(f"Removing {len(to_drop)} correlated features")
            X = X.drop(columns=to_drop)
        
        return X
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance DataFrame."""
        if not self.feature_scores:
            return pd.DataFrame()
        
        df = pd.DataFrame({
            'feature': list(self.feature_scores.keys()),
            'score': list(self.feature_scores.values())
        })
        return df.head(top_n)
    
    def get_support(self) -> np.ndarray:
        """Get feature support mask."""
        if self.selector:
            return self.selector.get_support()
        return np.array([True] * len(self.selected_features))


class EnsembleFeatureSelector:
    """Ensemble feature selection using multiple methods."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.methods = self.config.get('methods', [
            'mutual_info', 'f_statistic', 'random_forest', 'logistic_l1'
        ])
        self.k_best = self.config.get('k_best', 50)
        self.vote_threshold = self.config.get('vote_threshold', 2)
        
        self.feature_votes: Dict[str, int] = {}
        self.selected_features: List[str] = []
    
    def fit(self, X: pd.DataFrame, y: pd.Series, task_type: str = 'classification') -> 'EnsembleFeatureSelector':
        """Fit ensemble selector."""
        all_scores = {}
        
        for method in self.methods:
            try:
                scores = self._get_method_scores(X, y, method, task_type)
                all_scores[method] = scores
                logger.info(f"{method}: computed scores for {len(scores)} features")
            except Exception as e:
                logger.warning(f"{method} failed: {e}")
        
        # Voting
        self.feature_votes = {}
        for method, scores in all_scores.items():
            top_features = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:self.k_best]
            for feat in top_features:
                self.feature_votes[feat] = self.feature_votes.get(feat, 0) + 1
        
        # Select features with enough votes
        self.selected_features = [
            feat for feat, votes in self.feature_votes.items()
            if votes >= self.vote_threshold
        ]
        
        # If too few, lower threshold
        if len(self.selected_features) < self.k_best // 2:
            self.selected_features = sorted(
                self.feature_votes.keys(), 
                key=lambda x: self.feature_votes[x], 
                reverse=True
            )[:self.k_best]
        
        logger.info(f"Ensemble selected {len(self.selected_features)} features")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform feature matrix."""
        available = [f for f in self.selected_features if f in X.columns]
        return X[available]
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series, task_type: str = 'classification') -> pd.DataFrame:
        """Fit and transform."""
        self.fit(X, y, task_type)
        return self.transform(X)
    
    def _get_method_scores(
        self, X: pd.DataFrame, y: pd.Series, method: str, task_type: str
    ) -> Dict[str, float]:
        """Get scores for a specific method."""
        if method == 'mutual_info':
            if task_type == 'classification':
                scores = mutual_info_classif(X, y, random_state=42)
            else:
                scores = mutual_info_regression(X, y, random_state=42)
            return dict(zip(X.columns, scores))
        
        elif method == 'f_statistic':
            if task_type == 'classification':
                scores, _ = f_classif(X, y)
            else:
                scores, _ = f_regression(X, y)
            return dict(zip(X.columns, scores))
        
        elif method == 'random_forest':
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            return dict(zip(X.columns, rf.feature_importances_))
        
        elif method == 'logistic_l1':
            lr = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=42, max_iter=1000)
            lr.fit(X, y)
            return dict(zip(X.columns, np.abs(lr.coef_[0])))
        
        else:
            raise ValueError(f"Unknown method: {method}")


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    method: str = 'mutual_info',
    k: int = 50,
    task_type: str = 'classification',
) -> Tuple[pd.DataFrame, List[str]]:
    """Convenience function for feature selection."""
    if method == 'ensemble':
        selector = EnsembleFeatureSelector({'k_best': k})
    else:
        selector = FeatureSelector({'method': method, 'k_best': k})
    
    X_selected = selector.fit_transform(X, y, task_type)
    return X_selected, selector.selected_features


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=100, n_informative=20, random_state=42)
    X_df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(100)])
    y_series = pd.Series(y)
    
    selector = FeatureSelector({'k_best': 20})
    X_sel = selector.fit_transform(X_df, y_series)
    print(f"Selected: {len(X_sel.columns)} features")
    print(selector.get_feature_importance(10))
