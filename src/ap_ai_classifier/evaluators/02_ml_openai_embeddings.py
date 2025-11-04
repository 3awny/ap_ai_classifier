"""
Comprehensive evaluation of OpenAI embedding-based classifiers using MultiOutputClassifier.
Predicts NOMINAL, DEPARTMENT, and TC simultaneously for each invoice line item.

Uses MultiOutputClassifier with different base estimators:
1. LogisticRegression - Fast, interpretable, good for linear relationships
2. RandomForestClassifier - Handles non-linear relationships, robust to overfitting

Includes minimal hyperparameter tuning for imbalanced high-dimensional classification.
"""
import numpy as np
import pandas as pd
import logging
import time
from sklearn.metrics import f1_score
from ap_ai_classifier.core import load_and_prepare_data, encode_labels, stratified_random_split
from ap_ai_classifier.embeddings import create_embedding_model
from ap_ai_classifier.models import MlClassifier
from ap_ai_classifier.models.llm.metrics import calculate_sample_wise_f1
from ap_ai_classifier.config import BASE_DIR

logger = logging.getLogger(__name__)


def evaluate_without_tuning(X_train, X_test, y_train, y_test, base_classifier_type, method_name):
    """Evaluate MultiOutputClassifier with baseline (default) parameters using pre-computed embeddings."""
    logger.info(f"\n{'='*60}")
    logger.info(f"{method_name} (Baseline)")
    logger.info(f"{'='*60}\n")
    
    start_time = time.time()
    
    logger.info(f"📦 Train: {len(X_train)}, Test: {len(X_test)} | Training...")
    
    # Use MlClassifier with embeddings + multi-output
    clf = MlClassifier(
        feature_type='embeddings',
        use_multioutput=True,
        base_classifier=base_classifier_type,
        tune_hyperparameters=False,
        random_state=42,
        n_jobs=-1
    )
    
    # Train
    base_classifier_name = base_classifier_type.replace('_', ' ').title()
    logger.info(f"🎯 Training MultiOutputClassifier ({base_classifier_name})...")
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    
    # Calculate metrics per field
    fields = ['NOMINAL', 'DEPARTMENT', 'TC']
    results = {}
    overall_f1_scores = []
    
    for idx, field in enumerate(fields):
        f1_macro = f1_score(y_test[:, idx], y_pred[:, idx], average='macro', zero_division=0)
        accuracy = (y_pred[:, idx] == y_test[:, idx]).mean()
        overall_f1_scores.append(f1_macro)
        results[field] = {
            'f1_macro': f1_macro,
            'accuracy': accuracy
        }
        logger.info(f"   {field:<12}: F1-Macro: {f1_macro:.4f} | Accuracy: {accuracy:.4f}")
    
    # Overall average F1-macro
    avg_f1_macro = np.mean(overall_f1_scores)
    elapsed = time.time() - start_time
    
    # Calculate sample-wise F1 (accounts for partial correctness)
    # Convert predictions and true labels to tuples for sample-wise calculation
    true_labels = [(y_test[i, 0], y_test[i, 1], y_test[i, 2]) for i in range(len(y_test))]
    pred_labels = [(y_pred[i, 0], y_pred[i, 1], y_pred[i, 2]) for i in range(len(y_pred))]
    # Note: These are encoded labels (integers), but calculate_sample_wise_f1 works with any comparable values
    sample_wise_f1 = calculate_sample_wise_f1(true_labels, pred_labels)
    
    marker = " ✅" if avg_f1_macro >= 0.70 else ""
    logger.info(f"\n📊 {method_name} (Baseline):")
    logger.info(f"   Average F1-Macro (field-wise average): {avg_f1_macro:.4f}{marker}")
    logger.info(f"   Sample-wise F1 (accounts for partial correctness): {sample_wise_f1:.4f}")
    logger.info(f"   Time: {elapsed:.1f}s")
    
    return {
        'method': method_name + ' (Baseline)',
        'avg_f1_macro': avg_f1_macro,
        'sample_wise_f1': sample_wise_f1,  # New metric for partial correctness
        'nominal_f1': results['NOMINAL']['f1_macro'],
        'department_f1': results['DEPARTMENT']['f1_macro'],
        'tc_f1': results['TC']['f1_macro'],
        'time_seconds': elapsed,
        'best_params': None,  # No tuning
        'per_field_results': results,
        'is_baseline': True
    }


def tune_and_evaluate_multioutput_classifier(X_train, X_test, y_train, y_test, base_classifier_type, method_name, n_iter=20):
    """Tune hyperparameters and evaluate MultiOutputClassifier with given base estimator using pre-computed embeddings."""
    logger.info(f"\n{'='*60}")
    logger.info(f"{method_name}")
    logger.info(f"{'='*60}\n")
    
    start_time = time.time()
    
    logger.info(f"📦 Train: {len(X_train)}, Test: {len(X_test)} | Tuning hyperparameters...")
    
    # Use MlClassifier with embeddings + multi-output + tuning
    clf = MlClassifier(
        feature_type='embeddings',
        use_multioutput=True,
        base_classifier=base_classifier_type,
        tune_hyperparameters=True,
        cv_folds=3,
        n_iter=n_iter,
        random_state=42,
        n_jobs=-1
    )
    
    # Train (tuning happens inside fit)
    logger.info(f"🔍 Tuning hyperparameters ({n_iter} iterations)...")
    tune_start = time.time()
    clf.fit(X_train, y_train)
    tune_time = time.time() - tune_start
    
    logger.info(f"   ✅ Tuning complete ({tune_time:.1f}s)")
    if clf.best_params_:
        logger.info(f"   Best params: {clf.best_params_}")
    
    # Evaluate
    y_pred = clf.predict(X_test)
    
    # Calculate metrics per field
    fields = ['NOMINAL', 'DEPARTMENT', 'TC']
    results = {}
    overall_f1_scores = []
    
    for idx, field in enumerate(fields):
        f1_macro = f1_score(y_test[:, idx], y_pred[:, idx], average='macro', zero_division=0)
        accuracy = (y_pred[:, idx] == y_test[:, idx]).mean()
        overall_f1_scores.append(f1_macro)
        results[field] = {
            'f1_macro': f1_macro,
            'accuracy': accuracy
        }
        logger.info(f"   {field:<12}: F1-Macro: {f1_macro:.4f} | Accuracy: {accuracy:.4f}")
    
    # Overall average F1-macro
    avg_f1_macro = np.mean(overall_f1_scores)
    elapsed = time.time() - start_time
    
    # Calculate sample-wise F1 (accounts for partial correctness)
    # Convert predictions and true labels to tuples for sample-wise calculation
    true_labels = [(y_test[i, 0], y_test[i, 1], y_test[i, 2]) for i in range(len(y_test))]
    pred_labels = [(y_pred[i, 0], y_pred[i, 1], y_pred[i, 2]) for i in range(len(y_pred))]
    # Note: These are encoded labels (integers), but calculate_sample_wise_f1 works with any comparable values
    sample_wise_f1 = calculate_sample_wise_f1(true_labels, pred_labels)
    
    marker = " ✅" if avg_f1_macro >= 0.70 else ""
    logger.info(f"\n📊 {method_name}:")
    logger.info(f"   Average F1-Macro (field-wise average): {avg_f1_macro:.4f}{marker}")
    logger.info(f"   Sample-wise F1 (accounts for partial correctness): {sample_wise_f1:.4f}")
    logger.info(f"   Time: {elapsed:.1f}s (tuning: {tune_time:.1f}s)")
    
    return {
        'method': method_name,
        'avg_f1_macro': avg_f1_macro,
        'sample_wise_f1': sample_wise_f1,  # New metric for partial correctness
        'nominal_f1': results['NOMINAL']['f1_macro'],
        'department_f1': results['DEPARTMENT']['f1_macro'],
        'tc_f1': results['TC']['f1_macro'],
        'time_seconds': elapsed,
        'best_params': clf.best_params_,
        'per_field_results': results,
        'is_baseline': False
    }


def main(tune_hyperparameters=False):
    """Run comprehensive evaluation of OpenAI embedding-based approaches with MultiOutputClassifier."""
    logger.info("\n" + "="*60)
    tuning_label = " (with Hyperparameter Tuning)" if tune_hyperparameters else ""
    logger.info(f"OpenAI Embeddings + MultiOutputClassifier Evaluation{tuning_label}")
    logger.info("="*60 + "\n")
    
    # Load and prepare data
    df, label_vocab = load_and_prepare_data()
    
    logger.info(f"\n📊 Dataset: {len(df)} samples")
    logger.info(f"   NOMINAL classes: {len(label_vocab['nominal'])}")
    logger.info(f"   DEPARTMENT classes: {len(label_vocab['department'])}")
    logger.info(f"   TC classes: {len(label_vocab['tax_code'])}")
    
    # Generate embeddings ONCE for model features
    logger.info(f"\n🔮 Generating embeddings for model features...")
    embedding_model = create_embedding_model()
    logger.info(f"   Embedding Model: OpenAI {embedding_model.config.model_name}")
    logger.info(f"   Embedding Dimension: {embedding_model.config.embedding_dimension}D")
    logger.info(f"   Max Concurrency: {embedding_model.config.max_concurrency} threads")
    X_text = df["model_text"].tolist()
    fields = ['NOMINAL', 'DEPARTMENT', 'TC']
    y_all, encoders, decoders = encode_labels(df, fields)
    
    # Split data using stratified random split (preserves class distribution and training diversity)
    logger.info(f"\n🔮 Creating stratified random split (preserves class distribution)...")
    train_idx, test_idx = stratified_random_split(df, test_size=0.2, random_state=42)
    
    # Convert pandas Index to numpy array for indexing
    train_idx_array = df.index.get_indexer(train_idx)
    test_idx_array = df.index.get_indexer(test_idx)
    
    # Generate embeddings for train/test sets (using same approach as SemanticIndex for consistency)
    X_train_text = [X_text[i] for i in train_idx_array]
    X_test_text = [X_text[i] for i in test_idx_array]
    
    batch_size = 100
    train_batches = (len(X_train_text) + batch_size - 1) // batch_size
    test_batches = (len(X_test_text) + batch_size - 1) // batch_size
    total_batches = train_batches + test_batches
    logger.info(f"   Generating {total_batches} batches ({embedding_model.config.max_concurrency} threads)...")
    
    try:
        logger.info(f"   🔮 Building embeddings for training set ({train_batches} batches)...")
        X_train_embeddings = embedding_model.batch_get_embeddings_parallel(
            X_train_text, batch_size=batch_size
        )
        logger.info(f"   🔮 Building embeddings for test set ({test_batches} batches)...")
        X_test_embeddings = embedding_model.batch_get_embeddings_parallel(
            X_test_text, batch_size=batch_size
        )
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Interrupted by user")
        raise
    except Exception as e:
        logger.error(f"\n❌ Error during embedding generation: {e}")
        raise
    
    X_train = np.array(X_train_embeddings)
    X_test = np.array(X_test_embeddings)
    y_train = y_all[train_idx_array]
    y_test = y_all[test_idx_array]
    
    logger.info(f"✅ Stratified split created: Train {len(X_train)}, Test {len(X_test)}")
    logger.info(f"   (Reusing same embeddings and split for all evaluations)\n")
    
    # Evaluate different base classifiers
    results = []
    
    # Always run baseline first (using pre-computed embeddings)
    baseline_lr = evaluate_without_tuning(
        X_train, X_test, y_train, y_test,
        'logistic_regression',
        "MultiOutputClassifier with LogisticRegression"
    )
    results.append(baseline_lr)
    
    baseline_rf = evaluate_without_tuning(
        X_train, X_test, y_train, y_test,
        'random_forest',
        "MultiOutputClassifier with RandomForestClassifier"
    )
    results.append(baseline_rf)
    
    if tune_hyperparameters:
        # Run tuning versions (using same pre-computed embeddings)
        logger.info("\n" + "="*60)
        logger.info("HYPERPARAMETER TUNING")
        logger.info("="*60)
        
        # 1. LogisticRegression with hyperparameter tuning
        tuned_lr = tune_and_evaluate_multioutput_classifier(
            X_train, X_test, y_train, y_test,
            'logistic_regression',
            "MultiOutputClassifier with LogisticRegression (Tuned)",
            n_iter=20
        )
        results.append(tuned_lr)
        
        # 2. RandomForestClassifier with hyperparameter tuning
        tuned_rf = tune_and_evaluate_multioutput_classifier(
            X_train, X_test, y_train, y_test,
            'random_forest',
            "MultiOutputClassifier with RandomForestClassifier (Tuned)",
            n_iter=20
        )
        results.append(tuned_rf)
        
        # Compare baseline vs tuned for each classifier
        logger.info("\n" + "="*60)
        logger.info("BASELINE vs TUNED COMPARISON")
        logger.info("="*60 + "\n")
        
        # Compare LogisticRegression
        lr_baseline = baseline_lr
        lr_tuned = tuned_lr
        improvement = lr_tuned['avg_f1_macro'] - lr_baseline['avg_f1_macro']
        improvement_pct = (improvement / lr_baseline['avg_f1_macro'] * 100) if lr_baseline['avg_f1_macro'] > 0 else 0
        
        logger.info(f"LogisticRegression:")
        logger.info(f"   Baseline: F1-Macro = {lr_baseline['avg_f1_macro']:.4f}")
        logger.info(f"   Tuned:    F1-Macro = {lr_tuned['avg_f1_macro']:.4f}")
        if improvement > 0:
            logger.info(f"   ✅ Improvement: +{improvement:.4f} (+{improvement_pct:.2f}%)")
        elif improvement < 0:
            logger.warning(f"   ⚠️  Regression: {improvement:.4f} ({improvement_pct:.2f}%)")
        else:
            logger.info(f"   ➡️  No change")
        
        # Compare RandomForest
        rf_baseline = baseline_rf
        rf_tuned = tuned_rf
        improvement = rf_tuned['avg_f1_macro'] - rf_baseline['avg_f1_macro']
        improvement_pct = (improvement / rf_baseline['avg_f1_macro'] * 100) if rf_baseline['avg_f1_macro'] > 0 else 0
        
        logger.info(f"\nRandomForestClassifier:")
        logger.info(f"   Baseline: F1-Macro = {rf_baseline['avg_f1_macro']:.4f}")
        logger.info(f"   Tuned:    F1-Macro = {rf_tuned['avg_f1_macro']:.4f}")
        if improvement > 0:
            logger.info(f"   ✅ Improvement: +{improvement:.4f} (+{improvement_pct:.2f}%)")
        elif improvement < 0:
            logger.warning(f"   ⚠️  Regression: {improvement:.4f} ({improvement_pct:.2f}%)")
        else:
            logger.info(f"   ➡️  No change")
    
    # Summary comparison
    logger.info("\n" + "="*60)
    logger.info("OVERALL COMPARISON - MultiOutputClassifier Approaches")
    logger.info("="*60 + "\n")
    
    logger.info(f"{'Method':<55} {'Field F1':>10} {'Sample F1':>10} {'Time':>10}")
    logger.info("-" * 87)
    
    for r in results:
        marker = " 🎯" if r['avg_f1_macro'] >= 0.70 else ""
        baseline_indicator = " (Baseline)" if r.get('is_baseline', False) else ""
        sample_f1 = r.get('sample_wise_f1', 0.0)
        logger.info(f"{r['method']:<55} {r['avg_f1_macro']:>10.4f}{marker} {sample_f1:>10.4f} {r['time_seconds']:>9.1f}s")
    
    logger.info("-" * 87)
    
    # Show best parameters for tuned models
    if tune_hyperparameters:
        logger.info("\n📊 Best Hyperparameters for Tuned Models:")
        logger.info("-" * 92)
        for r in results:
            if r.get('best_params') and not r.get('is_baseline', False):
                logger.info(f"\n{r['method']}:")
                for param, value in r['best_params'].items():
                    # Remove 'estimator__' prefix for cleaner display
                    clean_param = param.replace('estimator__', '')
                    logger.info(f"   {clean_param:<25}: {value}")
        
        logger.info("\n" + "-" * 92)
    
    best = max(results, key=lambda x: x['avg_f1_macro'])
    best_type = "Tuned" if not best.get('is_baseline', False) else "Baseline"
    logger.info(f"\n🏆 Best Method: {best['method']} ({best_type})")
    logger.info(f"   Average F1-Macro (field-wise average): {best['avg_f1_macro']:.4f}")
    logger.info(f"   Sample-wise F1 (accounts for partial correctness): {best.get('sample_wise_f1', 0.0):.4f}")
    
    # If tuning was done, show comparison with baseline
    if tune_hyperparameters:
        baseline_results = [r for r in results if r.get('is_baseline', False)]
        tuned_results = [r for r in results if not r.get('is_baseline', False)]
        best_baseline = max(baseline_results, key=lambda x: x['avg_f1_macro'])
        best_tuned = max(tuned_results, key=lambda x: x['avg_f1_macro'])
        
        improvement = best_tuned['avg_f1_macro'] - best_baseline['avg_f1_macro']
        improvement_pct = (improvement / best_baseline['avg_f1_macro'] * 100) if best_baseline['avg_f1_macro'] > 0 else 0
        
        logger.info(f"\n📈 Tuning Impact:")
        logger.info(f"   Best Baseline: {best_baseline['avg_f1_macro']:.4f}")
        logger.info(f"   Best Tuned:    {best_tuned['avg_f1_macro']:.4f}")
        if improvement > 0:
            logger.info(f"   ✅ Tuning improved by +{improvement:.4f} (+{improvement_pct:.2f}%)")
        elif improvement < 0:
            logger.warning(f"   ⚠️  Tuning regressed by {improvement:.4f} ({improvement_pct:.2f}%)")
        else:
            logger.info(f"   ➡️  No improvement from tuning")
    
    if best['avg_f1_macro'] >= 0.70:
        logger.info(f"   ✅ Exceeds target by {(best['avg_f1_macro'] - 0.70)*100:.1f}%")
    else:
        logger.warning(f"   ⚠️  Gap to target: {(0.70 - best['avg_f1_macro'])*100:.1f}%")
    
    # Show per-field breakdown for best method
    logger.info(f"\n📊 Per-Field Breakdown (Best Method):")
    for field in ['NOMINAL', 'DEPARTMENT', 'TC']:
        f1 = best['per_field_results'][field]['f1_macro']
        logger.info(f"   {field:<12}: F1-Macro: {f1:.4f}")
    
    # Generate error report CSV for best method
    logger.info("\n" + "="*60)
    logger.info("ERROR ANALYSIS (Best Method)")
    logger.info("="*60)
    
    # Re-run prediction for best method to get detailed error report
    # Find the best method's classifier type
    best_method_name = best['method']
    if 'LogisticRegression' in best_method_name:
        best_classifier_type = 'logistic_regression'
    elif 'RandomForest' in best_method_name:
        best_classifier_type = 'random_forest'
    else:
        best_classifier_type = 'logistic_regression'  # Default
    
    is_tuned = not best.get('is_baseline', False)
    
    # Recreate classifier and get predictions
    clf_for_report = MlClassifier(
        feature_type='embeddings',
        use_multioutput=True,
        base_classifier=best_classifier_type,
        tune_hyperparameters=is_tuned,
        cv_folds=3,
        n_iter=20,
        random_state=42,
        n_jobs=-1
    )
    clf_for_report.fit(X_train, y_train)
    y_pred_best = clf_for_report.predict(X_test)
    
    # Reuse existing decoders (from line 205) - they use field names as keys
    # Decoders structure: {'NOMINAL': {0: 'label1', 1: 'label2', ...}, ...}
    
    # Get test DataFrame rows using test indices
    df_test = df.loc[test_idx].reset_index(drop=True)
    
    error_rows = []
    for i, (_, row) in enumerate(df_test.iterrows()):
        # Decode predictions using field names as keys (NOMINAL, DEPARTMENT, TC)
        # Add defensive checks in case model predicts unseen class indices
        try:
            pred_nom_decoded = decoders['NOMINAL'][y_pred_best[i, 0]]
        except KeyError:
            logger.warning(f"Warning: Prediction index {y_pred_best[i, 0]} not in NOMINAL decoder. Using 'UNKNOWN'.")
            pred_nom_decoded = 'UNKNOWN'
        
        try:
            pred_dep_decoded = decoders['DEPARTMENT'][y_pred_best[i, 1]]
        except KeyError:
            logger.warning(f"Warning: Prediction index {y_pred_best[i, 1]} not in DEPARTMENT decoder. Using 'UNKNOWN'.")
            pred_dep_decoded = 'UNKNOWN'
        
        try:
            pred_tc_decoded = decoders['TC'][y_pred_best[i, 2]]
        except KeyError:
            logger.warning(f"Warning: Prediction index {y_pred_best[i, 2]} not in TC decoder. Using 'UNKNOWN'.")
            pred_tc_decoded = 'UNKNOWN'
        
        true_nom = row['NOMINAL']
        true_dep = row['DEPARTMENT']
        true_tc = row['TC']
        
        # Check if any field has an error
        has_error = (pred_nom_decoded != true_nom) or (pred_dep_decoded != true_dep) or (pred_tc_decoded != true_tc)
        
        if has_error:
            error_rows.append({
                'detail': row['DETAIL'],
                'supplier': row['SUPPLIER'],
                'true_nominal': true_nom,
                'pred_nominal': pred_nom_decoded,
                'true_department': true_dep,
                'pred_department': pred_dep_decoded,
                'true_tc': true_tc,
                'pred_tc': pred_tc_decoded,
            })
    
    if error_rows:
        error_df = pd.DataFrame(error_rows)
        method_safe_name = best_method_name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', '').lower()
        error_report_path = BASE_DIR / "artifacts" / f"error_report_ml_openai_{method_safe_name}.csv"
        error_df.to_csv(error_report_path, index=False)
        logger.info(f"\n📊 Error analysis saved to: {error_report_path}")
        logger.info(f"   Found {len(error_df)} misclassified items out of {len(y_test)} total")
    else:
        logger.info(f"\n✅ No errors found! All {len(y_test)} items classified correctly.")
    
    logger.info("\n" + "="*60)
    
    return {
        'experiment': 'ML with OpenAI Embeddings (MultiOutputClassifier)' + (' (Tuned)' if tune_hyperparameters else ''),
        'best_method': best['method'],
        'best_f1_macro': best['avg_f1_macro'],
        'best_sample_wise_f1': best.get('sample_wise_f1', 0.0),  # New metric for partial correctness
        'nominal_f1': best['nominal_f1'],
        'department_f1': best['department_f1'],
        'tc_f1': best['tc_f1'],
        'all_results': results
    }


if __name__ == "__main__":
    main()
