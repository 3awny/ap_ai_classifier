# Evaluation Results

This folder contains the complete evaluation results showing the chronological progression through three experiments.

## 📋 Results Timeline

### 1. ML Baseline (TF-IDF)
**File:** [`01_ml_baseline_tfidf.md`](01_ml_baseline_tfidf.md)

- **Approach:** TF-IDF + Logistic Regression (keyword-based features)
- **Result:** F1-Macro **0.7548** ✅ (exceeds 0.70 target by 5.5%)
- **Best Model:** Tuned (hyperparameter tuning improved by +0.16%)
- **Error Rate:** 10.5% (53/503 errors)
- **Speed:** <1s (instant)
- **Cost:** $0 (no API)

### 2. ML with OpenAI Embeddings
**File:** [`02_ml_openai_embeddings.md`](02_ml_openai_embeddings.md)

- **Approach:** OpenAI 1536-dim embeddings + MultiOutputClassifier
- **Result:** F1-Macro **0.7798** ✅ (exceeds 0.70 target by 8.0%)
- **Best Model:** LogisticRegression Baseline (default hyperparameters optimal)
- **Error Rate:** 15.3% (77/503 errors)
- **Speed:** 3.5s training + instant inference
- **Cost:** ~$0.01 one-time
- **Improvement:** +3.3% over TF-IDF (stronger TC performance: 0.9246)

### 3. LLM vs ML Final Comparison
**File:** [`03_llm_vs_ml_final.md`](03_llm_vs_ml_final.md)

- **Approach:** RAG with few-shot learning (semantic retrieval + LLM reasoning)
- **Results:**
  - **gpt-5-mini:** F1-Macro **0.8077** ✅ (Winner - Best Accuracy)
  - **gpt-4.1-mini:** F1-Macro **0.8019** ✅ (Balanced speed/accuracy)
- **Error Rates:** 8.5% (gpt-5-mini), 9.1% (gpt-4.1-mini)
- **Speed:** 0.52-0.77s per sample
- **Cost:** $1-10 per 10K samples
- **Improvement:** +3.6% over ML OpenAI (better NOMINAL/DEPARTMENT performance)

---

## 🏆 Final Model Comparison

| Model | F1-Macro | Error Rate | Speed | Cost (10K) | Status |
|-------|----------|------------|-------|------------|--------|
| **gpt-5-mini** | **0.8077** 🏆 | **8.5%** | 0.77s | $5-10 | Best accuracy |
| **gpt-4.1-mini** | **0.8019** | **9.1%** | 0.52s | $1-2 | Balanced |
| **ML OpenAI** | **0.7798** | 15.3% | Instant | $0.01 | Best production |
| **ML TF-IDF** | **0.7548** | 10.5% | <1s | $0 | No API needed |

**All models exceed 0.70 target** ✅

---

## 💡 Recommendations

### Production Strategy

**Hybrid Approach** (Recommended):
1. **ML OpenAI Embeddings** for initial classification (fast, cheap)
2. Route **low-confidence predictions** (<0.7) to **gpt-4.1-mini**
3. Use **gpt-5-mini** only for critical/disputed cases
4. **Result:** 90%+ handled by ML, 5-10% by LLM

### When to Use Each Model

| Model | Best For | Key Advantage |
|-------|----------|---------------|
| **gpt-5-mini** | Accuracy-critical applications | Best F1 (0.8077), explainable reasoning |
| **gpt-4.1-mini** | Balanced production | Good F1 (0.8019), faster than gpt-5-mini |
| **ML OpenAI** | High-volume processing | 74× faster, 100× cheaper |
| **ML TF-IDF** | Offline/embedded systems | No API dependency, zero cost |

---

## 📊 Key Insights

1. **LLM models** achieve 30-40% fewer errors than ML (8.5-9.1% vs 10.5-15.3%)
2. **ML models** are 74-110× faster and 100-1000× cheaper
3. **Field complexity matters:**
   - NOMINAL (29 classes): LLM excels (+27.5% over ML)
   - TC (8 classes): ML excels (+14.8% over LLM)
4. **Error patterns** consistent across all models: see [`../artifacts/README.md`](../artifacts/README.md)

---

*All evaluations: 503 test samples (80/20 stratified random split)*  
*Target: F1-Macro ≥ 0.70*
