# Deliverable 2 - Explanations

---

## Task 1: Data Balancing (Regression)

### Target Variable: After-tax Income — Summary Statistics

| Statistic | Value |
|-----------|-------|
| Count | 64,930 (train) / 16,233 (test) |
| Mean | $46,863 |
| Median | $39,602 |
| Std Dev | $41,607 |
| Min | -$1,595,080 |
| Max | $980,990 |
| Skewness | 1.8687 (right-skewed) |
| Kurtosis | 89.3532 (extremely heavy tails) |
| Negative values | 265 |
| Zero values | 1,459 |

### Outlier Analysis (IQR Method)

| Measure | Value |
|---------|-------|
| Q1 | $21,875 |
| Q3 | $62,570 |
| IQR | $40,695 |
| Lower Bound | -$39,168 |
| Upper Bound | $123,612 |
| Outliers Below | 32 |
| Outliers Above | 2,316 |
| **Total Outliers** | **2,348 (3.6%)** |

### Why This Is Imbalance
- Mean ($46,863) >> Median ($39,602) → right skew, high earners pull the mean up
- 97% of data in $-39K to $123K range → model learns this dense region well
- 3.6% outliers (mostly above $123K) → model poorly predicts extreme earners
- Kurtosis = 89.35 → extremely heavy tails compared to normal distribution (kurtosis = 3)
- RMSE ($36,505) >> MAE ($19,099) → large prediction errors on outliers inflate RMSE

### Transformations Applied

| # | Technique | Applied To | Purpose |
|---|-----------|-----------|---------|
| 1 | Log Transform | Target (y) | Compress right tail, reduce skewness |
| 2 | Square-root Transform | Target (y) | Moderate compression, less aggressive than log |
| 3 | Box-Cox Transform | Target (y) | Optimal power transform (data-driven lambda = 0.9570) |
| 4 | Robust Scaling | Features (X) | Scale features using median/IQR, reduce outlier influence |

### Skewness Before vs After

| Transformation | Skewness | Change from Original |
|---------------|----------|---------------------|
| Original | 1.8687 | — |
| Log | -194.4714 | Overcorrected (negative extreme) |
| Square Root | -5.9490 | Overcorrected (negative) |
| Box-Cox (λ=0.957) | 1.6326 | Slight reduction |
| Robust Scaling | 1.8687 | No change (applied to features only) |

### Why Skewness Results Look This Way
- **Log & Sqrt overcorrected** → the large shift value ($1,595,081) needed to handle negative incomes dominates the transformation; log(1,595,081 + income) compresses the actual income variation into a tiny range relative to the shift
- **Box-Cox λ ≈ 0.957** → lambda close to 1.0 means the data is nearly linear already after shifting; minimal transformation applied
- **Robust Scaling** → only scales features (X), not target (y) → target skewness unchanged

### Transformation Impact on Model Performance (Random Forest, default params)

| Transformation | R² | Adj R² | RMSE ($) | MAE ($) | MAPE (%) |
|---------------|-----|--------|----------|---------|----------|
| **Original** | **0.2628** | **0.2623** | **36,505** | **19,099** | **220.37** |
| Log | 0.1302 | 0.1296 | 39,652 | 19,412 | 218.94 |
| Square Root | 0.2609 | 0.2604 | 36,551 | 19,082 | 218.62 |
| Box-Cox | 0.2628 | 0.2623 | 36,505 | 19,099 | 220.65 |
| **Robust Scaling** | **0.2629** | **0.2624** | **36,502** | **19,097** | **220.31** |

### Why Transformations Did Not Improve Results Significantly
- **Log Transform hurt R² badly (0.13)** → the massive shift ($1.6M) to handle negatives makes log nearly linear in the income range; model learns distorted relationships, inverse transform amplifies errors
- **Sqrt Transform slight decrease** → same shift problem but milder; squared inverse magnifies small prediction errors
- **Box-Cox ≈ no change** → λ=0.957 ≈ 1.0, meaning the optimal transform is essentially no transform after shifting
- **Robust Scaling marginal gain** → Random Forest is a tree-based model, invariant to feature scaling (splits based on rank order, not magnitude); the tiny improvement is noise
- **Root cause**: the extreme negative minimum (-$1.6M) forces a massive shift, which overwhelms the actual income distribution and makes log/sqrt ineffective

### Trade-offs

| Technique | Benefit | Risk |
|-----------|---------|------|
| Log Transform | Reduces skewness for naturally positive data | Requires shift for negatives → distorts relationships; worst R² here |
| Sqrt Transform | Milder compression than log | Same shift problem; squared inverse amplifies errors |
| Box-Cox | Data-driven optimal lambda | λ≈1 when shift dominates → essentially no transform |
| Robust Scaling | Outlier-resistant feature scaling | No effect on tree-based models (split-invariant to monotonic scaling) |

### Evaluation Metrics Used

| Metric | What It Measures | Value (Best Model) | Interpretation |
|--------|-----------------|-------------------|----------------|
| MAE | Average absolute error | $19,097 | On average, predictions off by ~$19K |
| RMSE | Root mean squared error | $36,502 | Large errors (outliers) inflate this to ~$36.5K |
| R² | Variance explained | 0.2629 | Model explains ~26% of income variation |
| Adj R² | R² penalized for features | 0.2624 | Minimal penalty (11 features, 64K rows) |
| MAPE | % error relative to actual | 220% | Inflated by near-zero incomes in denominator |

---

## Task 2: Hyperparameter Explanation

### Model 1: Decision Tree Regressor

| Hyperparameter | What It Controls | Impact on Complexity | Overfitting vs Underfitting | Why It Matters for This Dataset |
|----------------|-----------------|---------------------|---------------------------|-------------------------------|
| `max_depth` | Maximum tree depth | Higher depth → more complex | Deep trees overfit; shallow trees underfit | Best = 10: captures age × education interactions without memorizing 81K individual rows |
| `min_samples_split` | Minimum samples to split a node | Lower → more splits → more complex | Low values → overfit to small groups | Best = 2: dataset large enough (64K) to support fine splits without noise |
| `min_samples_leaf` | Minimum samples in a leaf node | Lower → smaller leaves → more complex | Low values → overfit; high values → oversmooth | Best = 10: prevents leaves with 1-2 extreme incomes ($500K+) from dominating predictions |

### Model 2: Random Forest Regressor

| Hyperparameter | What It Controls | Impact on Complexity | Overfitting vs Underfitting | Why It Matters for This Dataset |
|----------------|-----------------|---------------------|---------------------------|-------------------------------|
| `n_estimators` | Number of trees in forest | More trees → more computation, marginal gain | More trees reduce variance (less overfit) | Best = 100: sufficient averaging for 64K rows; 200 adds computation with minimal gain |
| `max_depth` | Maximum depth per tree | Deeper → captures more patterns | Deep + many trees can still overfit | Best = 10: matches DT finding; limits specificity per tree |
| `min_samples_leaf` | Minimum samples per leaf | Larger → simpler trees | High values prevent overfitting | Best = 5: slightly finer than DT (ensemble averaging compensates for variance) |

### Model 3: Gradient Boosting Regressor

| Hyperparameter | What It Controls | Impact on Complexity | Overfitting vs Underfitting | Why It Matters for This Dataset |
|----------------|-----------------|---------------------|---------------------------|-------------------------------|
| `n_estimators` | Number of boosting stages | More stages → more complex | Too many → overfit; too few → underfit | Best = 200: income errors need many iterative corrections to capture subtle patterns |
| `max_depth` | Depth of each base tree | Deeper → captures more interactions | Shallow (3-5) works best with boosting | Best = 5: captures age × education × employment interactions |
| `learning_rate` | Step size per update | Lower → needs more estimators | Low rate + many trees = best generalization | Best = 0.1: balanced step size; not too aggressive, not too slow |
| `min_samples_leaf` | Minimum samples per leaf | Larger → more regularization | High values prevent overfitting | Best = 5: prevents individual boosting stages from fitting to extreme income outliers |

---

## Task 3: GridSearchCV Results

### Best Parameters Found

| Model | Best Parameters | CV R² | Test R² | Combos Tested |
|-------|----------------|-------|---------|---------------|
| Decision Tree | max_depth=10, min_samples_split=2, min_samples_leaf=10 | 0.3404 | 0.3141 | 36 |
| Random Forest | n_estimators=100, max_depth=10, min_samples_leaf=5 | 0.3543 | 0.3252 | 18 |
| **Gradient Boosting** | **n_estimators=200, max_depth=5, learning_rate=0.1, min_samples_leaf=5** | **0.3629** | **0.3384** | **36** |

### Why These Parameters Won
- **max_depth=10 (DT, RF)** → depth 10 captures enough feature interactions without memorizing noise
- **max_depth=5 (GB)** → boosting builds many shallow trees; depth 5 per stage + 200 stages = effective depth much higher
- **min_samples_leaf=5-10** → prevents leaves from fitting to handful of extreme incomes
- **learning_rate=0.1** → moderate correction speed; 0.05 too slow for 200 stages, 0.2 overfits
- **n_estimators=100-200** → enough trees for stable ensemble averaging / error correction

---

## Task 4: Model Evaluation & Best Model Selection

### Full Comparison Table

| Model | Stage | R² | Adj R² | RMSE ($) | MAE ($) | MAPE (%) |
|-------|-------|-----|--------|----------|---------|----------|
| Linear Regression | D1 Baseline | 0.2358 | 0.2353 | 37,166 | 21,228 | 277.14 |
| Decision Tree | D1 Baseline | 0.1418 | 0.1412 | 39,388 | 20,496 | 266.37 |
| Random Forest | D1 Baseline | 0.2628 | 0.2623 | 36,505 | 19,099 | 220.37 |
| Decision Tree (Tuned) | D2 GridSearchCV | 0.3141 | 0.3136 | 35,211 | 18,170 | 209.47 |
| Random Forest (Tuned) | D2 GridSearchCV | 0.3252 | 0.3248 | 34,925 | 17,974 | 210.39 |
| **Gradient Boosting (Tuned)** | **D2 GridSearchCV** | **0.3384** | **0.3380** | **34,581** | **17,832** | **214.49** |

### D1 → D2 Improvement (Random Forest)

| Metric | D1 Baseline | D2 Tuned | Change |
|--------|-------------|----------|--------|
| R² | 0.2628 | 0.3252 | +0.0624 (+23.7%) |
| RMSE | $36,505 | $34,925 | -$1,580 (-4.3%) |
| MAE | $19,099 | $17,974 | -$1,125 (-5.9%) |

### D1 → D2 Improvement (Decision Tree)

| Metric | D1 Baseline | D2 Tuned | Change |
|--------|-------------|----------|--------|
| R² | 0.1418 | 0.3141 | +0.1723 (+121.5%) |
| RMSE | $39,388 | $35,211 | -$4,177 (-10.6%) |
| MAE | $20,496 | $18,170 | -$2,326 (-11.3%) |

### Why Each Model Performed the Way It Did

| Model | Rank | Why |
|-------|------|-----|
| **Gradient Boosting (Tuned)** | **#1** | Sequential error correction → each new tree focuses on what previous trees got wrong; 200 stages × depth 5 = captures complex income patterns |
| Random Forest (Tuned) | #2 | Ensemble averaging of 100 independent trees reduces variance; max_depth=10 prevents overfitting |
| Decision Tree (Tuned) | #3 | Single tree, but depth capping (10) and leaf size (10) prevent overfitting → huge jump from D1 baseline (+121% R²) |
| Random Forest (D1) | #4 | Good defaults (100 trees, unlimited depth), but unlimited depth causes some overfitting |
| Linear Regression | #5 | Assumes linear relationship between coded categories and income; cannot model interactions (age × education) |
| Decision Tree (D1) | #6 | Unlimited depth → memorized training data → worst generalization to test set |

### Why D1 Decision Tree Improved the Most in D2
- D1: unlimited depth → tree grew to hundreds of levels, memorized training data
- D2: max_depth=10, min_samples_leaf=10 → forced generalization
- R² jumped from 0.14 → 0.31 (121% improvement) — the largest gain of any model
- Shows that **Decision Trees are highly sensitive to depth control**

### Why All Models Cap Around R² ≈ 0.34
- **Missing key predictors**: occupation, industry, actual hours worked, years of experience are not in this dataset
- **Categorical granularity**: age grouped into 15 bins (not exact age), education into 4 levels → limits precision
- **Income variance within groups**: two people with same age group, education, and province can earn $20K or $200K depending on factors not in the data
- **Target has extreme outliers**: min = -$1.6M, max = $981K → no model can reliably predict these from demographic categories alone

### Data Characteristics Influencing Results
- **All 11 features are categorical/ordinal** → tree-based models split naturally on categories; linear regression treats them as numeric (less appropriate)
- **Non-linear interactions** → income depends on age × education × employment status combinations → tree/boosting models capture this, linear models cannot
- **Right-skewed target (skew=1.87)** → outlier incomes inflate RMSE for all models
- **Large sample size (64,930 train)** → benefits ensemble methods; enough data for reliable 5-fold CV

### Evaluation Trade-offs
- **R² vs MAE**: Gradient Boosting has best R² (0.338) but not best MAPE (214%) → explains most variance overall but relative % error is slightly worse than Decision Tree (209%)
- **RMSE vs MAE gap**: RMSE ($34,581) ≈ 2× MAE ($17,832) → confirms large errors on outlier predictions persist even after tuning
- **CV R² vs Test R²**: small gap (0.36 vs 0.34 for GB) → tuning produced models that generalize well, not just fit training data
- **Complexity vs performance**: Gradient Boosting trains slower (200 sequential stages) but gains +1.3% R² over Random Forest → worth the tradeoff for this dataset size
