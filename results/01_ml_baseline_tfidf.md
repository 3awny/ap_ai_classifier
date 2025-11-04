# Experiment 1: ML Baseline (TF-IDF)

## Summary

**TF-IDF + Logistic Regression** evaluated with hyperparameter tuning on stratified random split.

**Result: ✅ Exceeds Target**
- **Average F1-Macro: 0.7548** (Target: 0.70)
- **Sample-wise F1: 0.9490** ✅
- **Exceeds target by 5.5%**

---

## Results

### Baseline Performance

| Field | F1-Macro | Accuracy | Status |
|-------|----------|----------|--------|
| NOMINAL | 0.6038 | 95.83% | ❌ Below target |
| DEPARTMENT | 0.8272 | 92.25% | ✅ Exceeds target |
| TC | 0.8296 | 96.62% | ✅ Exceeds target |
| **AVERAGE** | **0.7535** | **94.90%** | **✅ Exceeds target** |

**Sample-wise F1**: 0.9490 ✅

### Tuned Performance

| Field | F1-Macro | Accuracy | Status |
|-------|----------|----------|--------|
| NOMINAL | 0.6006 | 95.43% | ❌ Below target |
| DEPARTMENT | 0.8316 | 92.25% | ✅ Exceeds target |
| TC | 0.8320 | 97.22% | ✅ Exceeds target |
| **AVERAGE** | **0.7548** | **94.97%** | **✅ Exceeds target** |

**Tuning Impact**: +0.0012 (+0.16%) improvement

---

## Configuration

### Dataset
- **Total Samples**: 2,515
- **Train**: 2,012 (80%)
- **Test**: 503 (20%)
- **Split Type**: Stratified random (fallback to random due to rare classes)
- **Classes**: 29 NOMINAL, 10 DEPARTMENT, 8 TC

### TF-IDF Vectorizer
- `max_features`: 5000
- `ngram_range`: (1, 2)
- `min_df`: 1
- `max_df`: 1.0
- `sublinear_tf`: False

### Logistic Regression
- `solver`: lbfgs
- `C`: 1.0
- `max_iter`: 200

---

## Hyperparameter Tuning

### Best Tuned Parameters

**NOMINAL** (CV F1: 0.6724):
- `clf__C`: 2.0
- `tfidf__ngram_range`: (1, 3)
- `tfidf__max_df`: 1.0
- `tfidf__sublinear_tf`: False

**DEPARTMENT** (CV F1: 0.8369):
- `clf__C`: 5.0
- `tfidf__ngram_range`: (1, 3)
- `tfidf__min_df`: 2

**TC** (CV F1: 0.8276):
- `clf__C`: 5.0
- `tfidf__ngram_range`: (1, 2)
- `tfidf__max_df`: 0.9
- `tfidf__sublinear_tf`: True

---

## Error Analysis

**Model Used**: Tuned (F1-Macro: 0.7548)  
**Errors**: 53 misclassified items out of 503 (10.5% error rate)  
**Error Report**: `../artifacts/error_report_ml_tfidf_tuned.csv`

### Top Error Patterns

1. **Department Confusion** (73.6% of errors): KITCHEN → HOUSEKEEPING (10×), MAINTENANCE → CAPITAL (5×)
2. **Tax Code Confusion** (26.4% of errors): T9 → T3 (4×), T3 → T6 (3×)
3. **Nominal Confusion** (43.4% of errors): Cleaning → Miscellaneous Expenses (3×)

**For detailed confusion matrices and pattern analysis**, see [`../artifacts/README.md`](../artifacts/README.md)

---

## Key Findings

1. **NOMINAL challenges** (F1: 0.6006):
   - Struggles with service-type nominals (Professional Fees, Cleaning)
   - Supplier-specific patterns not captured (e.g., SUPPLIER 23 = Gas)
   - 29 classes create high complexity

2. **DEPARTMENT & TC perform well** (F1: 0.8316, 0.8320):
   - Fewer classes (10, 8) easier to distinguish
   - Strong correlation with text patterns

3. **Hyperparameter tuning minimal impact** (+0.16%):
   - Baseline already well-optimized
   - Limited by TF-IDF's semantic understanding

---

## Conclusion

✅ **TF-IDF baseline meets target** (0.7548 vs 0.70)

**Strengths:**
- Fast training and inference
- No API dependencies
- Exceeds target by 5.5%

**Limitations:**
- Poor semantic understanding (keyword-based)
- Struggles with complex NOMINAL field (0.6006)
- Cannot capture supplier-specific patterns
- 10.5% error rate

**Next Step:** Experiment 2 evaluates OpenAI Embeddings for improved semantic understanding.

---

*Test Set: 503 samples (80/20 stratified random split)*  
*Best Model: Tuned (0.7548 F1-Macro)*
