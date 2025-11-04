from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

def semantic_holdout_split(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.Index, pd.Index]:
    """
    Cluster by semantic description, then group-split by cluster.
    This keeps clusters together and makes the test set more "novel".
    
    WARNING: Can create unrepresentative training set by removing diversity.
    Consider using stratified_random_split instead for better training diversity.
    """
    n_samples = embeddings.shape[0]
    n_clusters = max(5, int(n_samples ** 0.5))
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="euclidean",
        linkage="ward"
    )
    cluster_labels = clustering.fit_predict(embeddings)

    gss = GroupShuffleSplit(
        n_splits=1,
        train_size=1 - test_size,
        random_state=random_state
    )
    train_idx, test_idx = next(gss.split(df, groups=cluster_labels))
    return df.index[train_idx], df.index[test_idx]


def stratified_random_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify_by: str = 'NOMINAL',  # Stratify by most imbalanced field
) -> Tuple[pd.Index, pd.Index]:
    """
    Stratified random split that preserves class distribution.
    
    Advantages over semantic split:
    - Maintains training set diversity and representativeness
    - Preserves class distribution (reduces class imbalance)
    - Ensures exact test_size percentage
    - Better for training robust models
    
    Args:
        df: DataFrame with classification labels
        test_size: Proportion of test set (default: 0.2)
        random_state: Random seed for reproducibility
        stratify_by: Column name to stratify by (default: 'NOMINAL')
    
    Returns:
        Tuple of (train_indices, test_indices) as pandas Index
    """
    import warnings
    
    # Try multi-label stratification first (all 3 fields)
    try:
        stratify_label = df['NOMINAL'].astype(str) + '_' + df['DEPARTMENT'].astype(str) + '_' + df['TC'].astype(str)
        le = LabelEncoder()
        stratify_encoded = le.fit_transform(stratify_label)
        
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_state
        )
        train_idx, test_idx = next(sss.split(df, stratify_encoded))
        return df.index[train_idx], df.index[test_idx]
    
    except ValueError:
        # Multi-label stratification failed (some combinations have < 2 samples)
        # Fall back to single-field stratification on the most imbalanced field
        warnings.warn(
            f"Multi-label stratification failed. Falling back to single-field stratification on '{stratify_by}'. "
            f"Some rare class combinations may not be preserved in train/test split.",
            UserWarning
        )
        
        try:
            sss = StratifiedShuffleSplit(
                n_splits=1,
                test_size=test_size,
                random_state=random_state
            )
            train_idx, test_idx = next(sss.split(df, df[stratify_by]))
            return df.index[train_idx], df.index[test_idx]
        
        except ValueError:
            # Even single-field stratification failed
            # Fall back to simple random split
            warnings.warn(
                f"Stratification on '{stratify_by}' also failed (some classes have < 2 samples). "
                f"Using simple random split. Train/test sets may be imbalanced.",
                UserWarning
            )
            
            from sklearn.model_selection import ShuffleSplit
            ss = ShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            train_idx, test_idx = next(ss.split(df))
            return df.index[train_idx], df.index[test_idx]
