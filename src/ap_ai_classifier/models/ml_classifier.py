"""
ML Classifiers for AP classification.

SINGLE SOURCE OF TRUTH for all ML classification and hyperparameter tuning.

Supports:
1. TF-IDF + single-output classification (per field)
2. Pre-computed embeddings + multi-output classification (all fields simultaneously)

All hyperparameter tuning logic is centralized here:
- TF-IDF mode: Uses GridSearchCV (see _get_tfidf_param_grid, _fit_tfidf)
- Embeddings mode: Uses RandomizedSearchCV (see _get_embeddings_param_grid, _fit_embeddings)

Both evaluators (01_ml_tfidf_baseline.py and 02_ml_openai_embeddings.py) use this class,
ensuring consistent tuning behavior across all ML experiments.
"""

import numpy as np
import pandas as pd
import logging
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, make_scorer
from scipy.stats import randint
from typing import Optional, Dict, Any, Union, Type
import time

logger = logging.getLogger(__name__)


class MlClassifier:
    """
    ML classifier with support for TF-IDF (single-output) or pre-computed embeddings (multi-output).
    
    For TF-IDF: Takes text input, vectorizes it, and predicts single field.
    For embeddings: Takes pre-computed embeddings and can predict multiple fields simultaneously.
    """
    
    def __init__(
        self,
        feature_type: str = 'tfidf',  # 'tfidf' or 'embeddings'
        target_col: Optional[str] = None,  # Required for single-output (TF-IDF)
        use_multioutput: bool = False,  # True for multi-output classification
        base_classifier: Union[str, Type] = 'logistic_regression',  # 'logistic_regression' or 'random_forest' or class
        tune_hyperparameters: bool = False,
        cv_folds: int = 5,
        n_iter: int = 20,  # Number of iterations for RandomizedSearchCV
        n_jobs: int = -1,
        random_state: int = 42
    ):
        """
        Initialize ML classifier.
        
        Args:
            feature_type: 'tfidf' (text input) or 'embeddings' (pre-computed arrays)
            target_col: Target column name for single-output (e.g., 'NOMINAL', 'DEPARTMENT', 'TC')
            use_multioutput: If True, use MultiOutputClassifier for predicting all fields simultaneously
            base_classifier: Base classifier type ('logistic_regression', 'random_forest') or class
            tune_hyperparameters: Whether to perform hyperparameter tuning
            cv_folds: Number of CV folds for tuning
            n_jobs: Number of parallel jobs (-1 = use all cores)
            random_state: Random seed for reproducibility
        """
        self.feature_type = feature_type
        self.target_col = target_col
        self.use_multioutput = use_multioutput
        self.tune_hyperparameters = tune_hyperparameters
        self.cv_folds = cv_folds
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.random_state = random_state
        
        # Determine base classifier class
        if isinstance(base_classifier, str):
            if base_classifier == 'logistic_regression':
                self.base_classifier_class = LogisticRegression
            elif base_classifier == 'random_forest':
                self.base_classifier_class = RandomForestClassifier
            else:
                raise ValueError(f"Unknown base_classifier: {base_classifier}")
        else:
            self.base_classifier_class = base_classifier
        
        self.model = None
        self.best_params_: Optional[Dict[str, Any]] = None
        self.cv_best_score_: Optional[float] = None
        
        # Validate configuration
        if feature_type == 'tfidf' and use_multioutput:
            raise ValueError("TF-IDF mode doesn't support multi-output. Use embeddings instead.")
        if feature_type == 'tfidf' and target_col is None:
            raise ValueError("target_col is required for TF-IDF mode")
    
    def _create_base_classifier(self, **kwargs):
        """Create base classifier with default or custom parameters."""
        default_params = {
            'random_state': self.random_state,
            'n_jobs': self.n_jobs if hasattr(self.base_classifier_class, '__init__') else None,
        }
        
        if self.base_classifier_class == LogisticRegression:
            default_params.update({
                'max_iter': 3000,  # Dense embeddings need more iterations to converge
                'tol': 1e-3,
                'class_weight': 'balanced',  # Handle imbalanced classes (29 NOMINAL classes)
                'C': 1.0,
                'solver': 'lbfgs'
            })
        elif self.base_classifier_class == RandomForestClassifier:
            default_params.update({
                'n_estimators': 100,
                'class_weight': 'balanced',
                'max_depth': 10
            })
        
        default_params.update(kwargs)
        # Remove None values
        default_params = {k: v for k, v in default_params.items() if v is not None}
        
        return self.base_classifier_class(**default_params)
    
    def _create_pipeline(self) -> Pipeline:
        """Create TF-IDF pipeline (for single-output mode)."""
        if self.feature_type != 'tfidf':
            raise ValueError("Pipeline creation only supported for TF-IDF mode")
        
        return Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=1,
                max_df=1.0,
                sublinear_tf=False
            )),
            # Sparse TF-IDF features converge faster → 200 iter sufficient (vs 3000 for dense embeddings)
            ("clf", self._create_base_classifier(max_iter=200))
        ])
    
    def _get_tfidf_param_grid(self) -> Dict[str, list]:
        """
        Get hyperparameter grid for TF-IDF mode.
        
        Uses GridSearchCV because:
        - Small search space: ~192 combinations (3*2*2*2*2*4 = 192)
        - Discrete parameters only (no continuous ranges)
        - Fast to train (sparse TF-IDF features) → exhaustive search feasible
        """
        return {
            'tfidf__max_features': [5000, 7500, 10000],
            'tfidf__ngram_range': [(1, 2), (1, 3)],
            'tfidf__min_df': [1, 2],
            'tfidf__max_df': [0.9, 1.0],
            'tfidf__sublinear_tf': [True, False],
            'clf__C': [0.5, 1.0, 2.0, 5.0],
            'clf__penalty': ['l2'],
            'clf__solver': ['lbfgs'],
        }
    
    def _get_embeddings_param_grid(self, classifier_type: str) -> Dict[str, list]:
        """
        Get hyperparameter grid for embeddings mode.
        
        Uses RandomizedSearchCV because:
        - RandomForest uses continuous ranges (randint) → infinite combinations
        - Dense embeddings slower to train → need to limit iterations
        - RandomizedSearchCV samples n_iter=20 random combinations (vs exhaustive search)
        - More efficient for high-dimensional embeddings (1536-dim features)
        """
        if classifier_type == 'logistic_regression':
            return {
                'estimator__C': [0.01, 0.1, 1.0, 10.0, 100.0],
                'estimator__solver': ['lbfgs'],
                'estimator__class_weight': ['balanced', None],
                'estimator__max_iter': [2000, 3000],
                'estimator__tol': [1e-4, 1e-3]
            }
        elif classifier_type == 'random_forest':
            return {
                'estimator__n_estimators': randint(100, 300),
                'estimator__max_depth': [None, 10, 20, 30],
                'estimator__min_samples_split': randint(2, 11),
                'estimator__min_samples_leaf': randint(1, 5),
                'estimator__max_features': ['sqrt', 'log2'],
                'estimator__class_weight': ['balanced', None]
            }
        else:
            return {}
    
    def fit(self, X: Union[pd.Series, np.ndarray], y: Union[pd.Series, np.ndarray]):
        """
        Train the classifier.
        
        Args:
            X: Training features (text Series for TF-IDF, numpy array for embeddings)
            y: Training labels (Series for single-output, numpy array for multi-output)
        """
        if self.feature_type == 'tfidf':
            self._fit_tfidf(X, y)
        else:  # embeddings
            self._fit_embeddings(X, y)
        
        return self
    
    def _fit_tfidf(self, X: pd.Series, y: pd.Series):
        """Fit TF-IDF pipeline (single-output)."""
        base_pipeline = self._create_pipeline()
        
        if self.tune_hyperparameters:
            logger.info(f"   🔍 Performing grid search for {self.target_col}...")
            logger.info(f"      (This may take a few minutes)")
            
            # GridSearchCV: Exhaustive search over ~192 combinations (feasible for sparse TF-IDF)
            grid_search = GridSearchCV(
                base_pipeline,
                param_grid=self._get_tfidf_param_grid(),
                cv=self.cv_folds,
                scoring='f1_macro',
                n_jobs=self.n_jobs,
                verbose=1,  # Show progress: verbose=1 (basic), verbose=2 (each combination), verbose=3 (detailed per-fold)
                return_train_score=False
            )
            
            grid_search.fit(X, y)
            self.model = grid_search.best_estimator_
            self.best_params_ = grid_search.best_params_
            self.cv_best_score_ = grid_search.best_score_
            
            logger.info(f"      ✅ Best CV F1-Macro: {self.cv_best_score_:.4f}")
            logger.info(f"      📊 Best params: {self.best_params_}")
        else:
            self.model = base_pipeline
            self.model.fit(X, y)
    
    def _fit_embeddings(self, X: np.ndarray, y: np.ndarray):
        """Fit embeddings-based classifier (multi-output or single-output)."""
        base_classifier = self._create_base_classifier()
        
        if self.use_multioutput:
            clf = MultiOutputClassifier(base_classifier, n_jobs=self.n_jobs)
        else:
            clf = base_classifier
        
        if self.tune_hyperparameters:
            # Determine classifier type for param grid
            classifier_type = 'logistic_regression' if self.base_classifier_class == LogisticRegression else 'random_forest'
            param_dist = self._get_embeddings_param_grid(classifier_type)
            
            # Custom scorer for multi-output
            if self.use_multioutput:
                def multioutput_f1_macro_scorer(y_true, y_pred):
                    f1_scores = []
                    for i in range(y_true.shape[1]):
                        f1 = f1_score(y_true[:, i], y_pred[:, i], average='macro', zero_division=0)
                        f1_scores.append(f1)
                    return np.mean(f1_scores)
                scorer = make_scorer(multioutput_f1_macro_scorer, greater_is_better=True)
            else:
                scorer = 'f1_macro'
            
            logger.info(f"🔍 Tuning hyperparameters ({self.n_iter} iterations)...")
            # RandomizedSearchCV: Samples n_iter combinations (efficient for continuous params & dense embeddings)
            random_search = RandomizedSearchCV(
                estimator=clf,
                param_distributions=param_dist,
                n_iter=self.n_iter,
                cv=3,
                scoring=scorer,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbose=1  # Show progress: verbose=1 (basic), verbose=2 (each combination), verbose=3 (detailed per-fold)
            )
            
            random_search.fit(X, y)
            self.model = random_search.best_estimator_
            self.best_params_ = random_search.best_params_
            self.cv_best_score_ = random_search.best_score_
            
            logger.info(f"   ✅ Tuning complete")
            logger.info(f"   Best params: {self.best_params_}")
        else:
            self.model = clf
            self.model.fit(X, y)
    
    def predict(self, X: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        """Predict labels."""
        if self.model is None:
            raise ValueError("Classifier not fitted. Call fit() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X: Union[pd.Series, np.ndarray]):
        """Predict class probabilities."""
        if self.model is None:
            raise ValueError("Classifier not fitted. Call fit() first.")
        
        probs = self.model.predict_proba(X)
        
        if self.feature_type == 'tfidf':
            classes = self.model.named_steps['clf'].classes_
            return pd.DataFrame(probs, columns=classes)
        else:
            return probs


# Alias for backward compatibility
BaselineClassifier = MlClassifier
