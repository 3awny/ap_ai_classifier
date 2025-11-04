import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Union
from ap_ai_classifier.data import load_raw_df
from ap_ai_classifier.config import RAW_CSV_PATH

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9\-\s\.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def add_model_text_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["detail_norm"] = df["DETAIL"].apply(normalize_text)
    df["supplier_norm"] = df["SUPPLIER"].apply(normalize_text)
    df["model_text"] = (
        "detail: " + df["detail_norm"] +
        " | supplier: " + df["supplier_norm"]
    )
    return df

def build_label_vocab(df: pd.DataFrame):
    return {
        "nominal": sorted(df["NOMINAL"].dropna().unique().tolist()),
        "department": sorted(df["DEPARTMENT"].dropna().unique().tolist()),
        "tax_code": sorted(df["TC"].dropna().unique().tolist()),
    }

def load_and_prepare_data(csv_path: Optional[Path] = None) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Load and prepare data for evaluation.
    
    Args:
        csv_path: Optional path to CSV file (defaults to RAW_CSV_PATH)
        
    Returns:
        Tuple of (dataframe with model_text column, label_vocab dictionary)
    """
    df = load_raw_df(csv_path or RAW_CSV_PATH)
    df = add_model_text_column(df)
    label_vocab = build_label_vocab(df)
    return df, label_vocab

def encode_labels(df: pd.DataFrame, fields: List[str]) -> Tuple[np.ndarray, Dict[str, Dict[str, int]], Dict[str, Dict[int, str]]]:
    """
    Encode labels for all fields into numeric format for multi-output classification.
    
    Args:
        df: DataFrame with label columns
        fields: List of field names to encode (e.g., ['NOMINAL', 'DEPARTMENT', 'TC'])
        
    Returns:
        Tuple of:
        - y_encoded: numpy array of shape (n_samples, n_outputs)
        - encoders: Dict mapping field -> label_to_idx mapping
        - decoders: Dict mapping field -> idx_to_label mapping
    """
    encoders = {}
    decoded = {}
    y_encoded = []
    
    for field in fields:
        # Drop NaN values before encoding to avoid KeyError
        unique_labels = sorted(df[field].dropna().unique())
        if len(unique_labels) == 0:
            raise ValueError(f"Field {field} has no valid (non-NaN) labels to encode")
        
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}
        encoders[field] = label_to_idx
        decoded[field] = idx_to_label
        
        # Handle NaN values by mapping them to a special index
        # Check if we have NaN values and warn if so
        nan_count = df[field].isna().sum()
        if nan_count > 0:
            import warnings
            warnings.warn(f"Field {field} has {nan_count} NaN values. These will be encoded as -1. "
                         f"Consider filtering them out before encoding.")
        
        y_field = np.array([
            label_to_idx[label] if pd.notna(label) else -1
            for label in df[field].values
        ])
        y_encoded.append(y_field)
    
    # Shape: (n_samples, n_outputs)
    y_encoded = np.column_stack(y_encoded)
    
    return y_encoded, encoders, decoded

def prepare_multilabel_data(
    df: pd.DataFrame,
    train_idx: Union[List[int], pd.Index],
    test_idx: Union[List[int], pd.Index]
) -> Tuple[List[str], List[str], List[Tuple[str, str, str]], pd.DataFrame, pd.DataFrame]:
    """
    Prepare multi-label data for training/evaluation from train/test indices.
    
    Args:
        df: DataFrame with model_text, NOMINAL, DEPARTMENT, TC columns
        train_idx: List or pandas Index of training indices (integer positions)
        test_idx: List or pandas Index of test indices (integer positions)
        
    Returns:
        Tuple of:
        - X_train: List of training text strings
        - X_test: List of test text strings
        - y_train_labels: List of (nominal, department, tc) tuples for training
        - df_train: Training DataFrame
        - df_test: Test DataFrame
    """
    # Convert pandas Index to list if needed
    if isinstance(train_idx, pd.Index):
        train_idx = df.index.get_indexer(train_idx).tolist()
    if isinstance(test_idx, pd.Index):
        test_idx = df.index.get_indexer(test_idx).tolist()
    
    # Validate inputs
    if len(df) == 0:
        raise ValueError("Cannot prepare data from empty DataFrame")
    
    if len(train_idx) == 0:
        raise ValueError("Cannot prepare data with empty train_idx")
    
    if len(test_idx) == 0:
        raise ValueError("Cannot prepare data with empty test_idx")
    
    # Validate index bounds
    max_train_idx = max(train_idx) if train_idx else -1
    max_test_idx = max(test_idx) if test_idx else -1
    if max_train_idx >= len(df) or max_test_idx >= len(df):
        raise IndexError(f"Index out of bounds: DataFrame has {len(df)} rows, "
                        f"but max train_idx={max_train_idx}, max test_idx={max_test_idx}")
    
    X_train = [df["model_text"].iloc[i] for i in train_idx]
    X_test = [df["model_text"].iloc[i] for i in test_idx]
    
    y_train_labels = [
        (df["NOMINAL"].iloc[i], df["DEPARTMENT"].iloc[i], df["TC"].iloc[i])
        for i in train_idx
    ]
    
    # Use iloc for integer position-based indexing
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)
    
    return X_train, X_test, y_train_labels, df_train, df_test

