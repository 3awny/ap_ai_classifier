"""
ML Baseline Evaluation - TF-IDF + Logistic Regression

Evaluates the baseline TF-IDF + LogisticRegression approach on all three fields:
- NOMINAL
- DEPARTMENT  
- TC (Tax Code)

NOTE: This is a multi-output classification problem - we predict NOMINAL, DEPARTMENT, and TC
simultaneously for each invoice line item. However, we train separate models per field to:
1. Allow field-specific hyperparameters and optimization
2. Handle different class distributions per field
3. Enable per-field analysis

This is Approach #1 from APPROACH_RATIONALE.md
"""
import sys
import logging
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from ap_ai_classifier.models.ml_classifier import BaselineClassifier
from ap_ai_classifier.core import load_and_prepare_data, stratified_random_split
from ap_ai_classifier.config import BASE_DIR
from ap_ai_classifier.models.llm.metrics import calculate_sample_wise_f1

logger = logging.getLogger(__name__)


def evaluate_field(df_train, df_test, field_col, tune_hyperparameters=False):
    """Evaluate baseline classifier on a single field using pre-computed train/test split."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating {field_col}")
    logger.info(f"{'='*60}")
    
    tuning_label = " (Baseline)" if not tune_hyperparameters else " (Tuned)"
    logger.info(f"📦 Train: {len(df_train)}, Test: {len(df_test)} | Training TF-IDF + LogisticRegression{tuning_label}...")
    
    clf = BaselineClassifier(target_col=field_col, tune_hyperparameters=tune_hyperparameters)
    clf.fit(df_train["model_text"], df_train[field_col])
    
    # Predict and evaluate
    y_pred = clf.predict(df_test["model_text"])
    y_true = df_test[field_col].values
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    
    logger.info(f"\n📊 {field_col} - TF-IDF + Logistic Regression{tuning_label}:")
    logger.info(f"   F1-Macro: {f1:.4f} | Accuracy: {acc:.4f} ({int(acc * len(y_true))}/{len(y_true)} correct)")
    
    result = {
        'field': field_col,
        'f1_macro': f1,
        'accuracy': acc,
        'test_size': len(df_test),
        'is_baseline': not tune_hyperparameters,
        'predictions': list(y_pred),  # Store predictions for sample-wise F1
        'true_labels': list(y_true)   # Store true labels for sample-wise F1
    }
    
    if tune_hyperparameters and clf.best_params_:
        result['best_params'] = clf.best_params_
    
    return result


def main(tune_hyperparameters=False):
    """Run baseline evaluation on all fields."""
    logger.info("\n" + "="*80)
    logger.info("ML BASELINE EVALUATION")
    logger.info("TF-IDF + Logistic Regression (Baseline)")
    logger.info("="*80)
    
    # Load and prepare data
    df, _ = load_and_prepare_data()
    
    # Log model information
    logger.info(f"\n📋 Model Configuration:")
    logger.info(f"   Vectorizer: TF-IDF (TfidfVectorizer)")
    logger.info(f"   Classifier: LogisticRegression")
    logger.info(f"   TF-IDF Parameters:")
    logger.info(f"      - max_features: 5000")
    logger.info(f"      - ngram_range: (1, 2)")
    logger.info(f"      - min_df: 1")
    logger.info(f"      - max_df: 1.0")
    logger.info(f"      - sublinear_tf: False")
    logger.info(f"   LogisticRegression Parameters:")
    logger.info(f"      - solver: lbfgs")
    logger.info(f"      - C: 1.0")
    logger.info(f"      - max_iter: 200")
    
    # Split data using stratified random split (preserves class distribution and training diversity)
    logger.info(f"\n🔮 Creating stratified random split (preserves class distribution)...")
    train_idx, test_idx = stratified_random_split(df, test_size=0.2, random_state=42)
    df_train = df.loc[train_idx].reset_index(drop=True)
    df_test = df.loc[test_idx].reset_index(drop=True)
    
    logger.info(f"✅ Split created: Train {len(df_train)}, Test {len(df_test)}")
    logger.info(f"   (Reusing same split for all fields and evaluations)\n")
    
    # Always run baseline first (using pre-computed split)
    baseline_results = []
    # Store predictions for sample-wise F1 calculation
    all_preds_nom, all_preds_dep, all_preds_tc = [], [], []
    all_trues_nom, all_trues_dep, all_trues_tc = [], [], []
    
    for field in ['NOMINAL', 'DEPARTMENT', 'TC']:
        result = evaluate_field(df_train, df_test, field, tune_hyperparameters=False)
        baseline_results.append(result)
        
        # Extract predictions and true labels from result
        if field == 'NOMINAL':
            all_preds_nom = result['predictions']
            all_trues_nom = result['true_labels']
        elif field == 'DEPARTMENT':
            all_preds_dep = result['predictions']
            all_trues_dep = result['true_labels']
        elif field == 'TC':
            all_preds_tc = result['predictions']
            all_trues_tc = result['true_labels']
    
    # Calculate sample-wise F1
    true_labels = list(zip(all_trues_nom, all_trues_dep, all_trues_tc))
    pred_labels = list(zip(all_preds_nom, all_preds_dep, all_preds_tc))
    sample_wise_f1 = calculate_sample_wise_f1(true_labels, pred_labels)
    
    # Summary for baseline
    logger.info("\n" + "="*80)
    logger.info("SUMMARY - ML BASELINE (TF-IDF + Logistic Regression)")
    logger.info("="*80)
    
    df_baseline = pd.DataFrame(baseline_results)
    logger.info(f"\n{df_baseline.to_string(index=False)}")
    
    baseline_avg_f1 = df_baseline['f1_macro'].mean()
    logger.info(f"\n🎯 Average F1-Macro (field-wise average): {baseline_avg_f1:.4f}")
    logger.info(f"📊 Sample-wise F1 (accounts for partial correctness): {sample_wise_f1:.4f}")
    
    if baseline_avg_f1 >= 0.70:
        logger.info(f"✅ PASS - Exceeds target of 0.70 (field-wise)")
    else:
        logger.warning(f"❌ FAIL - Below target of 0.70 (field-wise)")
    
    if sample_wise_f1 >= 0.70:
        logger.info(f"✅ PASS - Exceeds target of 0.70 (sample-wise)")
    else:
        logger.info(f"📝 Sample-wise F1: {sample_wise_f1:.4f} (shows partial correctness impact)")
    
    results = baseline_results.copy()
    
    # Initialize variables for best model selection
    tuned_results = []
    tuned_avg_f1 = None
    
    # If tuning is enabled, run tuning and compare (using same pre-computed split)
    if tune_hyperparameters:
        logger.info("\n" + "="*80)
        logger.info("HYPERPARAMETER TUNING")
        logger.info("="*80)
        
        tuned_results = []
        for field in ['NOMINAL', 'DEPARTMENT', 'TC']:
            logger.info(f"\n{'='*60}")
            logger.info(f"Tuning {field}")
            logger.info(f"{'='*60}")
            
            result = evaluate_field(df_train, df_test, field, tune_hyperparameters=True)
            tuned_results.append(result)
            results.append(result)
        
        # Compare baseline vs tuned
        logger.info("\n" + "="*80)
        logger.info("BASELINE vs TUNED COMPARISON")
        logger.info("="*80 + "\n")
        
        df_tuned = pd.DataFrame(tuned_results)
        tuned_avg_f1 = df_tuned['f1_macro'].mean()
        
        improvement = tuned_avg_f1 - baseline_avg_f1
        improvement_pct = (improvement / baseline_avg_f1 * 100) if baseline_avg_f1 > 0 else 0
        
        logger.info(f"Baseline Average F1-Macro: {baseline_avg_f1:.4f}")
        logger.info(f"Tuned Average F1-Macro:    {tuned_avg_f1:.4f}")
        
        if improvement > 0:
            logger.info(f"✅ Tuning improved by +{improvement:.4f} (+{improvement_pct:.2f}%)")
        elif improvement < 0:
            logger.warning(f"⚠️  Tuning regressed by {improvement:.4f} ({improvement_pct:.2f}%)")
        else:
            logger.info(f"➡️  No improvement from tuning")
        
        logger.info(f"\n📊 Best Hyperparameters (Tuned):")
        for r in tuned_results:
            if r.get('best_params'):
                logger.info(f"\n   {r['field']}:")
                for param, value in r['best_params'].items():
                    logger.info(f"      {param:<25}: {value}")
    
    # Generate error report CSV for best model
    logger.info("\n" + "="*80)
    logger.info("ERROR ANALYSIS" + (" (Best Model)" if tune_hyperparameters else ""))
    logger.info("="*80)
    
    # Determine which model to use for error report
    # If tuning was done, use the better of baseline vs tuned
    if tune_hyperparameters and tuned_avg_f1 is not None and tuned_avg_f1 > baseline_avg_f1:
        # Use tuned results (better performance)
        results_to_use = tuned_results
        model_label = "Tuned"
        logger.info(f"   Using Tuned model (F1-Macro: {tuned_avg_f1:.4f} vs Baseline: {baseline_avg_f1:.4f})")
    else:
        # Use baseline results (either no tuning done, or baseline is better)
        results_to_use = baseline_results
        model_label = "Baseline"
        if tune_hyperparameters and tuned_avg_f1 is not None:
            logger.info(f"   Using Baseline model (F1-Macro: {baseline_avg_f1:.4f} vs Tuned: {tuned_avg_f1:.4f})")
    
    # Collect all predictions and true labels from best results
    error_rows = []
    for i, (_, row) in enumerate(df_test.iterrows()):
        # Get predictions from best results - use next() with default to avoid StopIteration
        nom_result = next((r for r in results_to_use if r['field'] == 'NOMINAL'), None)
        dep_result = next((r for r in results_to_use if r['field'] == 'DEPARTMENT'), None)
        tc_result = next((r for r in results_to_use if r['field'] == 'TC'), None)
        
        # Validate that all results were found
        if nom_result is None or dep_result is None or tc_result is None:
            missing_fields = []
            if nom_result is None:
                missing_fields.append('NOMINAL')
            if dep_result is None:
                missing_fields.append('DEPARTMENT')
            if tc_result is None:
                missing_fields.append('TC')
            raise ValueError(f"Missing results for fields: {missing_fields}. Cannot generate error report.")
        
        # Validate index bounds
        if i >= len(nom_result['predictions']) or i >= len(dep_result['predictions']) or i >= len(tc_result['predictions']):
            raise IndexError(f"Index {i} out of bounds for predictions. Expected length {len(df_test)}, "
                           f"but predictions have lengths: NOMINAL={len(nom_result['predictions'])}, "
                           f"DEPARTMENT={len(dep_result['predictions'])}, TC={len(tc_result['predictions'])}")
        
        pred_nom = nom_result['predictions'][i]
        pred_dep = dep_result['predictions'][i]
        pred_tc = tc_result['predictions'][i]
        
        true_nom = nom_result['true_labels'][i]
        true_dep = dep_result['true_labels'][i]
        true_tc = tc_result['true_labels'][i]
        
        # Check if any field has an error
        has_error = (pred_nom != true_nom) or (pred_dep != true_dep) or (pred_tc != true_tc)
        
        if has_error:
            error_rows.append({
                'detail': row['DETAIL'],
                'supplier': row['SUPPLIER'],
                'true_nominal': true_nom,
                'pred_nominal': pred_nom,
                'true_department': true_dep,
                'pred_department': pred_dep,
                'true_tc': true_tc,
                'pred_tc': pred_tc,
            })
    
    if error_rows:
        error_df = pd.DataFrame(error_rows)
        # Generate filename based on which model is best
        if tune_hyperparameters and tuned_avg_f1 is not None and tuned_avg_f1 > baseline_avg_f1:
            error_report_filename = "error_report_ml_tfidf_tuned.csv"
        else:
            error_report_filename = "error_report_ml_tfidf_baseline.csv"
        error_report_path = BASE_DIR / "artifacts" / error_report_filename
        error_df.to_csv(error_report_path, index=False)
        logger.info(f"\n📊 Error analysis saved to: {error_report_path}")
        logger.info(f"   Model used: {model_label}")
        logger.info(f"   Found {len(error_df)} misclassified items out of {len(df_test)} total")
    else:
        logger.info(f"\n✅ No errors found! All {len(df_test)} items classified correctly.")
    
    logger.info("\n" + "="*80)
    
    return {
        'experiment': 'ML Baseline (TF-IDF)' + (' (with Tuning)' if tune_hyperparameters else ''),
        'avg_f1_macro': baseline_avg_f1 if not tune_hyperparameters else max(baseline_avg_f1, tuned_avg_f1),
        'sample_wise_f1': sample_wise_f1,  # New metric for partial correctness
        'nominal_f1': df_baseline[df_baseline['field'] == 'NOMINAL']['f1_macro'].iloc[0] if len(df_baseline[df_baseline['field'] == 'NOMINAL']) > 0 else 0.0,
        'department_f1': df_baseline[df_baseline['field'] == 'DEPARTMENT']['f1_macro'].iloc[0] if len(df_baseline[df_baseline['field'] == 'DEPARTMENT']) > 0 else 0.0,
        'tc_f1': df_baseline[df_baseline['field'] == 'TC']['f1_macro'].iloc[0] if len(df_baseline[df_baseline['field'] == 'TC']) > 0 else 0.0,
        'results': results
    }


if __name__ == "__main__":
    main()
