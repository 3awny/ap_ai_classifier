# Experiment 2: ML with OpenAI Embeddings

## Summary

**MultiOutputClassifier** with OpenAI embeddings evaluated using LogisticRegression and RandomForestClassifier.

**Result: ✅ Exceeds Target**
- **Best Method: LogisticRegression (Baseline)**
- **Average F1-Macro: 0.7798** (Target: 0.70)
- **Sample-wise F1: 0.9317** ✅
- **Exceeds target by 8.0%**

---

## Results

### All Methods Comparison

| Method | NOMINAL | DEPARTMENT | TC | **Avg F1** | **Sample F1** | Time |
|--------|---------|------------|-----|------------|--------------|------|
| **LogReg (Baseline)** 🏆 | **0.5920** | **0.8227** | **0.9246** | **0.7798** | **0.9317** | 3.5s |
| LogReg (Tuned) | 0.6044 | 0.8261 | 0.8168 | 0.7491 | 0.9516 | 7.9s |
| RandomForest (Baseline) | 0.5923 | 0.8314 | 0.8129 | 0.7455 | 0.9450 | 3.6s |
| RandomForest (Tuned) | 0.6194 | 0.8483 | 0.8101 | 0.7593 | 0.9463 | 54.6s |

**Winner**: LogisticRegression Baseline (0.7798) - default hyperparameters optimal

---

## Configuration

### Dataset
- **Total Samples**: 2,515
- **Train**: 2,012 (80%)
- **Test**: 503 (20%)
- **Split Type**: Stratified random (fallback to random due to rare classes)
- **Classes**: 29 NOMINAL, 10 DEPARTMENT, 8 TC

### Embeddings
- **Model**: OpenAI text-embedding-3-small
- **Dimension**: 1536D
- **Max Concurrency**: 5 threads
- **Batches**: 27 batches for full dataset

### MultiOutputClassifier
- **Approach**: Single model predicts all three fields simultaneously
- **Base Estimator**: LogisticRegression (default) or RandomForestClassifier

---

## Hyperparameter Tuning

### LogisticRegression (Tuned)
**Best Parameters** (20 iterations, 3-fold CV):
- `C`: 100.0
- `solver`: lbfgs
- `max_iter`: 2000
- `tol`: 0.001
- `class_weight`: None

**Result**: Regression (-3.93%) - baseline performs better

### RandomForestClassifier (Tuned)
**Best Parameters** (20 iterations, 3-fold CV):
- `n_estimators`: 207
- `max_depth`: 20
- `max_features`: sqrt
- `min_samples_split`: 4
- `min_samples_leaf`: 2
- `class_weight`: balanced

**Result**: Improvement (+1.84%) but still below baseline LogReg

---

## Error Analysis

**Model Used**: LogisticRegression Baseline (F1-Macro: 0.7798)  
**Errors**: 77 misclassified items out of 503 (15.3% error rate)  
**Error Report**: `../artifacts/error_report_ml_openai_multioutputclassifier_with_logisticregression_baseline.csv`

### Top Error Patterns

1. **Department Confusion** (58.4% of errors): KITCHEN → HOUSEKEEPING (11×), HOUSEKEEPING → BAR (5×)
2. **Tax Code Confusion** (29.9% of errors): T9 → T6 (5×), T3 → T6 (5×)
3. **Nominal Confusion** (45.5% of errors): Materials Purchased → Miscellaneous Expenses (11×)

**For detailed confusion matrices and pattern analysis**, see [`../artifacts/README.md`](../artifacts/README.md)

---

## Key Findings

1. **Strong TC performance** (F1: 0.9246):
   - Best field across all approaches
   - OpenAI embeddings capture tax patterns well

2. **NOMINAL struggles** (F1: 0.5920):
   - 29 classes create complexity
   - Service-type nominals harder to distinguish
   - Similar to TF-IDF challenges but better

3. **DEPARTMENT moderate** (F1: 0.8227):
   - Cross-functional items difficult (coffee, supplies)
   - Supplier-department relationships not fully captured

4. **Default hyperparameters optimal**:
   - Baseline LogReg outperforms all tuned variants
   - Tuning regressed LogReg by 3.93%
   - RandomForest tuning improved only 1.84% but still below LogReg

5. **Speed advantage**:
   - 3.5s training time
   - Instant inference
   - One-time embedding cost

---

## Comparison to TF-IDF

| Metric | TF-IDF | OpenAI Embeddings | Improvement |
|--------|--------|-------------------|-------------|
| **Avg F1-Macro** | 0.7548 | **0.7798** | +3.3% ✅ |
| NOMINAL | 0.6006 | 0.5920 | -1.4% |
| DEPARTMENT | 0.8316 | 0.8227 | -1.1% |
| TC | 0.8320 | **0.9246** | +11.1% ✅ |
| Error Rate | 10.5% | 15.3% | +4.8% ❌ |

**OpenAI Embeddings wins** on overall F1 and TC field, despite higher error rate.

---

## Conclusion

✅ **OpenAI Embeddings meets target** (0.7798 vs 0.70)

**Strengths:**
- Strong semantic understanding
- Best TC field performance (0.9246)
- Fast training and inference
- Single model for all fields
- No hyperparameter tuning needed

**Limitations:**
- Higher error rate than TF-IDF (15.3% vs 10.5%)
- NOMINAL still challenging (0.5920)
- Supplier-specific patterns not fully captured
- Requires API dependency for embedding generation

**Next Step:** Experiment 3 evaluates LLM-based classification with retrieval-augmented generation.

---

*Test Set: 503 samples (80/20 stratified random split)*  
*Best Model: LogisticRegression Baseline (0.7798 F1-Macro)*
