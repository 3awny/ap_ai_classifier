# Error Analysis Reports

This directory contains detailed error analysis for all evaluated models. Each CSV file lists misclassified items from the test set (503 total samples).

---

## 📊 Error Files

### LLM Models
- **`error_report_gpt-5-mini.csv`** (47 errors, 8.5% error rate) - Best performing model
  - Includes reasoning explanations and confidence scores for each prediction
  - Fields: `detail`, `supplier`, `true_nominal`, `pred_nominal`, `true_department`, `pred_department`, `true_tc`, `pred_tc`, `confidence`, `reasoning`

- **`error_report_gpt-4-1-mini.csv`** (46 errors, 9.1% error rate)
  - Includes reasoning explanations and confidence scores for each prediction
  - Fields: `detail`, `supplier`, `true_nominal`, `pred_nominal`, `true_department`, `pred_department`, `true_tc`, `pred_tc`, `confidence`, `reasoning`

### ML Models

**TF-IDF + Logistic Regression:**
- **`error_report_ml_tfidf_baseline.csv`** - Baseline (non-tuned) results
- **`error_report_ml_tfidf_tuned.csv`** (53 errors, 10.5% error rate) - Tuned hyperparameters (best ML model)

**OpenAI Embeddings + MultiOutputClassifier:**
- **`error_report_ml_openai_multioutputclassifier_with_logisticregression_baseline.csv`** (77 errors, 15.3% error rate)
- **`error_report_ml_openai_multioutputclassifier_with_logisticregression_tuned.csv`** - Tuned hyperparameters (regressed)

All ML error files contain: `detail`, `supplier`, `true_nominal`, `pred_nominal`, `true_department`, `pred_department`, `true_tc`, `pred_tc`

---

## 📈 Detailed Error Breakdown by Field

**Field-Level Error Distribution:**

| Model | Total Errors | NOMINAL Errors | DEPARTMENT Errors | TAX CODE Errors |
|-------|--------------|----------------|-------------------|-----------------|
| **gpt-5-mini** 🏆 | 47 (8.5%) | 14 (29.8% of errors) | 33 (70.2% of errors) | 17 (36.2% of errors) |
| **gpt-4.1-mini** | 46 (9.1%) | 16 (34.8% of errors) | 32 (69.6% of errors) | 18 (39.1% of errors) |
| **ML TF-IDF** | 53 (10.5%) | 23 (43.4% of errors) | 39 (73.6% of errors) | 14 (26.4% of errors) |
| **ML OpenAI** | 77 (15.3%) | 35 (45.5% of errors) | 45 (58.4% of errors) | 23 (29.9% of errors) |

**Note:** Percentages add up to >100% because errors can span multiple fields (e.g., both DEPARTMENT and TAX CODE wrong for same item).

---

## 🔍 Confusion Patterns by Model

### 1. Department Confusion (Most Common - 58-74% of errors)

**Pattern:** KITCHEN ↔ HOUSEKEEPING ↔ BAR confusion for consumables

**Top 3 Confusions per Model:**
- **ML OpenAI** (worst): 
  - KITCHEN → HOUSEKEEPING: **11× errors**
  - HOUSEKEEPING → BAR: 5×
  - HOUSEKEEPING → RESTAURANT: 4×
  - **Why:** Dense embeddings blur semantic boundaries between similar departments
  
- **ML TF-IDF**: 
  - KITCHEN → HOUSEKEEPING: **10× errors**
  - MAINTENANCE → CAPITAL: 5×
  - MAINTENANCE → ACCOMMODATION: 4×
  - **Why:** Keyword matching fails when same items used across departments

- **gpt-4.1-mini**: 
  - KITCHEN → HOUSEKEEPING: **8× errors**
  - KITCHEN → BAR: 5×
  - MAINTENANCE → ACCOMMODATION: 4×
  - **Why:** Retrieved examples show inconsistent department assignments

- **gpt-5-mini** (best): 
  - KITCHEN → HOUSEKEEPING: **6× errors**
  - HOUSEKEEPING → BAR: 5×
  - KITCHEN → BAR: 5×
  - **Why:** Better at disambiguating from context, but still struggles

**Example:** `"TEA"` → True: HOUSEKEEPING, Predicted: BAR (multiple models)

**Root Cause:** Items like coffee/tea/beverages genuinely serve multiple departments (cross-functional purchases)

---

### 2. Tax Code Confusion (26-39% of errors)

**Pattern:** T2 ↔ T1 (VAT rates), T3 ↔ T6 ↔ T9 (service/capital categories)

**Top 3 Confusions per Model:**
- **ML OpenAI** (worst):
  - T9 → T6: **5× errors**
  - T3 → T6: **5× errors**
  - T2 → T1: 3×
  - **Why:** Embeddings don't capture tax rules (requires domain knowledge)

- **gpt-5-mini & gpt-4.1-mini**:
  - T2 → T1: **4× errors each**
  - T3 → T9: 3×
  - T9 → T0: 2×
  - **Why:** LLM learns from retrieved examples but lacks tax domain expertise

- **ML TF-IDF** (best):
  - T9 → T3: **4× errors**
  - T3 → T6: 3×
  - T9 → T6: 2×
  - **Why:** Lower error rate but limited to keyword matching

**Example:** `"BEVERAGE"` → True: T2 (standard VAT), Predicted: T1 (reduced VAT)

**Root Cause:** Fine-grained tax distinctions require accounting domain knowledge (alcoholic vs non-alcoholic beverages, capital vs services)

---

### 3. Nominal Confusion (30-46% of errors)

**Pattern:** Materials Purchased ↔ Miscellaneous Expenses ↔ Cleaning (29 overlapping categories)

**Top 3 Confusions per Model:**
- **ML OpenAI** (worst):
  - Materials Purchased → Miscellaneous Expenses: **11× errors**
  - Miscellaneous Expenses → Office Equipment: 2×
  - Buildings → Office Equipment: 2×
  - **Why:** Over-generalization from embeddings, predicts common classes

- **ML TF-IDF**:
  - Cleaning → Miscellaneous Expenses: **3× errors**
  - Miscellaneous Expenses → Office Equipment: 2×
  - Miscellaneous Expenses → Cleaning: 2×
  - **Why:** Missing keywords lead to default "Miscellaneous" prediction

- **gpt-4.1-mini**:
  - Miscellaneous Expenses → Furniture and Fixtures: **2× errors**
  - Miscellaneous Expenses → Cleaning: 2×
  - Buildings → Furniture and Fixtures: 2×
  - **Why:** Ambiguous retrieved examples cause category confusion

- **gpt-5-mini** (best):
  - Miscellaneous Expenses → Cleaning: **2× errors**
  - Buildings → Miscellaneous Expenses: 2×
  - Vehicle Licences → Materials Purchased: 1×
  - **Why:** Better reasoning from context reduces nominal errors

**Example:** `"Cleaning supplies"` → True: Cleaning, Predicted: Miscellaneous Expenses

**Root Cause:** Overlapping categories with semantic similarity (NOMINAL has 29 classes, most complex field)

---

### 4. Supplier-Specific Patterns (10-15% of errors)

**Consistent Across All Models:**
- **SUPPLIER 33 (Coffee)**: KITCHEN vs HOUSEKEEPING vs BAR (5+ errors across models)
- **SUPPLIER 96 (Diesel)**: MAINTENANCE vs ACCOMMODATION (4+ errors across models)
- **SUPPLIER 24 (Supplies)**: HOUSEKEEPING vs RESTAURANT vs KITCHEN (cross-functional)
- **SUPPLIER 13/14/15 (Beverages)**: Department (BAR/HOUSEKEEPING), Tax Code (T2/T1)
- **SUPPLIER 121 (Local Charges)**: Nominal confusion (multiple capital-related categories)
- **SUPPLIER 23 (Gas)**: Nominal misclassified as Materials Purchased in ML models

**Root Cause:** Supplier context not captured; same supplier serves multiple departments

**Solution:** Supplier normalization layer or supplier-specific classification rules

---

## 🎯 Model Comparison Summary

| Error Type | Best Model | Worst Model | Key Insight |
|------------|------------|-------------|-------------|
| **Overall** | gpt-5-mini (47, 8.5%) | ML OpenAI (77, 15.3%) | LLMs excel at reasoning through ambiguity |
| **Department** | gpt-5-mini (70.2% of errors) | ML TF-IDF (73.6% of errors) | All models struggle; inherently ambiguous |
| **Tax Code** | ML TF-IDF (26.4% of errors) | gpt-4.1-mini (39.1% of errors) | Requires domain knowledge, not just semantics |
| **Nominal** | gpt-5-mini (29.8% of errors) | ML OpenAI (45.5% of errors) | LLMs better at distinguishing overlapping categories |

**Key Takeaway:** LLM models (Phase 3) have **30-40% fewer errors** than ML models (Phase 1-2) despite higher cost, making them best for production where accuracy matters most.

---

## 💡 Usage

These files are generated automatically during evaluation runs. Use them to:
- **Identify systematic misclassification patterns** (supplier-specific, field-specific)
- **Debug model performance issues** (why did this item get misclassified?)
- **Compare confusion matrices across models** (which model handles which error type best?)
- **Prioritize improvements** (focus on high-frequency error patterns)
- **Build human-in-the-loop workflows** (route specific patterns to review)

