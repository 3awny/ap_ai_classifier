"""Metrics and evaluation utilities for LLM classifiers."""
from typing import Dict, Any
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
from sklearn.metrics import f1_score
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

logger = logging.getLogger(__name__)

def calculate_sample_wise_f1(true_labels: list, pred_labels: list) -> float:
    """
    Calculate sample-wise F1 score for multi-output classification.
    
    For each sample, calculates F1 considering all fields together.
    This gives partial credit for samples with some correct predictions.
    
    Args:
        true_labels: List of tuples (nominal, department, tax_code) for true labels
        pred_labels: List of tuples (nominal, department, tax_code) for predicted labels
        
    Returns:
        Sample-wise F1 score (0.0 to 1.0)
    """
    if len(true_labels) != len(pred_labels):
        raise ValueError("True and predicted labels must have same length")
    
    # For each sample, calculate precision and recall
    # Precision = correct predictions / total predictions
    # Recall = correct predictions / total true labels
    # For multi-output, both are the same: correct_fields / total_fields
    
    sample_f1_scores = []
    
    for true_tup, pred_tup in zip(true_labels, pred_labels):
        # Count correct predictions
        correct = sum(1 for t, p in zip(true_tup, pred_tup) if t == p)
        total_fields = len(true_tup)
        
        if total_fields == 0:
            continue
            
        # For multi-output classification with one label per field,
        # precision = recall = accuracy_per_sample
        precision = recall = correct / total_fields
        
        # F1 = 2 * (precision * recall) / (precision + recall)
        # When precision == recall, this simplifies to precision
        if precision == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        sample_f1_scores.append(f1)
    
    # Average F1 across all samples
    return np.mean(sample_f1_scores) if sample_f1_scores else 0.0

def evaluate_llm_classifier(clf, df_test: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluate LLM classifier with threading for parallel predictions.
    
    Args:
        clf: LLM classifier instance with predict_one() method
        df_test: Test DataFrame with model_text, NOMINAL, DEPARTMENT, TC columns
        
    Returns:
        Dictionary containing:
        - nominal_acc, nominal_f1: Accuracy and F1-macro for NOMINAL field
        - department_acc, department_f1: Accuracy and F1-macro for DEPARTMENT field
        - tax_code_acc, tax_code_f1: Accuracy and F1-macro for TC field
        - details: DataFrame with predictions and ground truth
    """
    if len(df_test) == 0:
        raise ValueError("Cannot evaluate on empty test set")
    
    results = {}
    results_lock = Lock()
    
    def process_sample(idx: int, row: pd.Series) -> tuple:
        """Process a single sample and return results."""
        try:
            item = row.to_dict()
            item["model_text"] = row["model_text"]
            # Normalize field names to lowercase for prompt template
            item["detail"] = row["DETAIL"]
            item["supplier"] = row["SUPPLIER"]
            item["net"] = row["NET"]
            item["vat"] = row["VAT"]
            
            out = clf.predict_one(item)
            
            return idx, {
                "pred_nominal": out["nominal"],
                "pred_department": out["department"],
                "pred_tc": out["tax_code"],
                "confidence": out["confidence"],
                "reasoning": out.get("reasoning", ""),
            }
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            return idx, {
                "pred_nominal": "",
                "pred_department": "",
                "pred_tc": "",
                "confidence": {},
                "reasoning": "",
            }
    
    # Use ThreadPoolExecutor for parallel processing
    max_workers = 5
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(process_sample, i, row): i
            for i, (_, row) in enumerate(df_test.iterrows())
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=len(df_test), desc="Evaluating") as pbar:
            for future in as_completed(future_to_idx):
                idx, pred = future.result()
                with results_lock:
                    results[idx] = pred
                pbar.update(1)
    
    # Sort results by original index
    preds_nom, preds_dep, preds_tc = [], [], []
    trues_nom, trues_dep, trues_tc = [], [], []
    rows = []
    
    for i, (_, row) in enumerate(df_test.iterrows()):
        pred = results[i]
        
        preds_nom.append(pred["pred_nominal"])
        preds_dep.append(pred["pred_department"])
        preds_tc.append(pred["pred_tc"])
        trues_nom.append(row["NOMINAL"])
        trues_dep.append(row["DEPARTMENT"])
        trues_tc.append(row["TC"])
        
        # Convert confidence dict to string for CSV serialization
        confidence_str = str(pred["confidence"]) if pred["confidence"] else "{}"
        
        row_data = {
            "detail": row["DETAIL"],
            "supplier": row["SUPPLIER"],
            "true_nominal": row["NOMINAL"],
            "pred_nominal": pred["pred_nominal"],
            "true_department": row["DEPARTMENT"],
            "pred_department": pred["pred_department"],
            "true_tc": row["TC"],
            "pred_tc": pred["pred_tc"],
            "confidence": confidence_str,
            "reasoning": pred.get("reasoning", ""),  # Ensure reasoning is always a string
        }
        
        rows.append(row_data)

    df_res = pd.DataFrame(rows)

    def acc(true, pred):
        return (pd.Series(true) == pd.Series(pred)).mean()
    
    def f1_macro(true, pred):
        return f1_score(true, pred, average='macro', zero_division=0)
    
    # Calculate sample-wise F1 (accounts for partial correctness)
    # This gives credit for samples with 2/3 or 1/3 correct labels
    true_labels = list(zip(trues_nom, trues_dep, trues_tc))
    pred_labels = list(zip(preds_nom, preds_dep, preds_tc))
    sample_wise_f1 = calculate_sample_wise_f1(true_labels, pred_labels)

    return {
        "nominal_acc": acc(trues_nom, preds_nom),
        "nominal_f1": f1_macro(trues_nom, preds_nom),
        "department_acc": acc(trues_dep, preds_dep),
        "department_f1": f1_macro(trues_dep, preds_dep),
        "tax_code_acc": acc(trues_tc, preds_tc),
        "tax_code_f1": f1_macro(trues_tc, preds_tc),
        "sample_wise_f1": sample_wise_f1,  # New metric: accounts for partial correctness
        "details": df_res,
    }

