# Deliverable 2 - Explanations

---

## Task 1: Data Balancing (Regression)

### Target Variable: After-tax Income

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

### Target Outlier Analysis (IQR Method)

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

### Feature Outlier Analysis (IQR Method)

| Feature | Q1 | Q3 | IQR | Outliers |
|---------|----|----|-----|----------|
| Economic family type | 21 | 24 | 3 | 22,331 (34.4%) |
| Census family composition | 3 | 4 | 1 | 17,715 (27.3%) |
| Annual labour force status | 1 | 3 | 2 | 2,146 (3.3%) |
| All other features | — | — | — | 0 (0%) |

### Why This Is Imbalance
- Target: Mean ($46,863) >> Median ($39,602) → right skew from high earners
- Target: 3.6% outliers, but kurtosis = 89.35 → extreme tails
- Features: Economic family type has 34.4% outliers → unequal distribution across family types
- Features: Census family composition has 27.3% outliers → certain compositions are rare
- Impact: models bias toward the dense middle range, underperform on tails and rare groups

### Transformations Applied to Features

| # | Technique | What It Does | Purpose |
|---|-----------|-------------|---------|
| 1 | Standardization (StandardScaler) | Centers to mean=0, std=1 | Puts all features on equal scale |
| 2 | Normalization (MinMaxScaler) | Scales to [0, 1] range | Equalizes feature ranges |
| 3 | Robust Scaling (RobustScaler) | Centers by median, scales by IQR | Reduces influence of feature outliers |
| 4 | Log Transform (log1p) | Compresses large values | Reduces feature skewness |

### Feature Skewness: Before vs After

| Feature | Original | Std | MinMax | Robust | Log |
|---------|----------|-----|--------|--------|-----|
| Major source of income | 0.89 | 0.89 | 0.89 | 0.89 | **0.49** |
| Person's age group | -0.26 | -0.26 | -0.26 | -0.26 | -0.66 |
| Flag - Employed | 0.52 | 0.52 | 0.52 | 0.52 | 0.52 |
| Education level | -0.37 | -0.37 | -0.37 | -0.37 | -0.79 |
| Annual labour force status | **1.20** | 1.20 | 1.20 | 1.20 | **0.82** |
| Age youngest in EF | 0.09 | 0.09 | 0.09 | 0.09 | -0.57 |
| Economic family type | **1.12** | 1.12 | 1.12 | 1.12 | **0.13** |
| Marital status | 0.58 | 0.58 | 0.58 | 0.58 | 0.43 |
| Age oldest in EF | -0.49 | -0.49 | -0.49 | -0.49 | -0.88 |
| Flag - High school | 1.82 | 1.82 | 1.82 | 1.82 | 1.82 |
| Census family composition | 0.59 | 0.59 | 0.59 | 0.59 | -0.55 |

**Key finding**: Standardization, Normalization, and Robust Scaling do NOT change skewness — they are linear transforms (shift + scale). Only **Log Transform** actually reduces skewness (non-linear compression of large values).

### Model Performance: Before vs After Feature Transformations

**Linear Regression:**

| Transformation | R² | RMSE ($) | MAE ($) |
|---------------|-----|----------|---------|
| Original | 0.2358 | 37,166 | 21,228 |
| Standardization | 0.2358 | 37,166 | 21,228 |
| Normalization | 0.2358 | 37,166 | 21,228 |
| Robust Scaling | 0.2358 | 37,166 | 21,228 |
| Log Transform | 0.2346 | 37,195 | 21,297 |

**Ridge Regression:**

| Transformation | R² | RMSE ($) | MAE ($) |
|---------------|-----|----------|---------|
| Original | 0.2358 | 37,166 | 21,228 |
| Standardization | 0.2358 | 37,166 | 21,228 |
| Normalization | 0.2358 | 37,166 | 21,228 |
| Robust Scaling | 0.2358 | 37,166 | 21,228 |
| Log Transform | 0.2346 | 37,195 | 21,297 |

**Random Forest:**

| Transformation | R² | RMSE ($) | MAE ($) |
|---------------|-----|----------|---------|
| Original | 0.2628 | 36,505 | 19,099 |
| Standardization | 0.2628 | 36,504 | 19,097 |
| Normalization | 0.2625 | 36,511 | 19,101 |
| Robust Scaling | 0.2629 | 36,502 | 19,097 |
| Log Transform | 0.2621 | 36,523 | 19,107 |

### Why Feature Transformations Had Minimal Effect
- **Linear Regression**: scaling features does not change predictions — OLS finds the same solution regardless of scale (coefficients adjust proportionally)
- **Ridge Regression**: alpha=1.0 regularization is too mild to create meaningful differences across scales
- **Random Forest**: tree-based models split on rank order of values, not magnitude → monotonic transforms (scaling) cannot change split decisions
- **Log Transform slightly worse**: non-linear compression changes the feature relationships; for ordinal-coded categories (1-15), log compression distorts the spacing between categories
- **Root cause**: all 11 features are categorical/ordinal with small integer values (1-44) → no continuous features with wide ranges or extreme outliers where scaling would matter

### Trade-offs

| Technique | Benefit | Risk |
|-----------|---------|------|
| Standardization | Equal weight to all features; helps gradient-based models | No effect on tree-based models; assumes normal distribution |
| Normalization | Bounded [0,1] range; works with any distribution | Sensitive to outliers in features (min/max shift) |
| Robust Scaling | Outlier-resistant (uses median/IQR) | Still no effect on tree-based models |
| Log Transform | Reduces skewness (non-linear) | Distorts ordinal spacing; can hurt model if relationship was linear |

### Evaluation Metrics

| Metric | What It Measures | Interpretation |
|--------|-----------------|----------------|
| MAE | Average absolute error | ~$19K average error for best model |
| RMSE | Root mean squared error | ~$36.5K — inflated by outlier predictions |
| R² | Variance explained | ~0.26 — model explains 26% of income variation |
| Adj R² | R² penalized for features | ~0.26 — minimal penalty (11 features, 64K rows) |
| MAPE | % error relative to actual | ~220% — inflated by near-zero incomes |

---

## Task 2: Hyperparameter Explanation

### Model 1: Decision Tree Regressor

| Hyperparameter | What It Controls | Impact on Complexity | Overfitting vs Underfitting | Why It Matters for This Dataset |
|----------------|-----------------|---------------------|---------------------------|-------------------------------|
| `max_depth` | Maximum tree depth | Higher → more complex | Deep → overfit; shallow → underfit | Best=10: captures age × education interactions without memorizing 81K rows |
| `min_samples_split` | Min samples to split a node | Lower → more splits → complex | Low → overfit to small groups | Best=2: dataset large enough (64K) to support fine splits |
| `min_samples_leaf` | Min samples in leaf node | Lower → smaller leaves → complex | Low → overfit; high → oversmooth | Best=10: prevents leaves with 1-2 extreme income values |

### Model 2: Random Forest Regressor

| Hyperparameter | What It Controls | Impact on Complexity | Overfitting vs Underfitting | Why It Matters for This Dataset |
|----------------|-----------------|---------------------|---------------------------|-------------------------------|
| `n_estimators` | Number of trees | More → more computation | More trees → less variance (less overfit) | Best=100: enough averaging; 200 adds cost with minimal gain |
| `max_depth` | Max depth per tree | Deeper → more patterns | Deep + many trees can overfit | Best=10: limits specificity per tree |
| `min_samples_leaf` | Min samples per leaf | Larger → simpler trees | High → prevents overfitting outliers | Best=5: slightly finer than DT (ensemble averaging compensates) |

### Model 3: Gradient Boosting Regressor

| Hyperparameter | What It Controls | Impact on Complexity | Overfitting vs Underfitting | Why It Matters for This Dataset |
|----------------|-----------------|---------------------|---------------------------|-------------------------------|
| `n_estimators` | Number of boosting stages | More → more complex | Too many → overfit; too few → underfit | Best=200: income errors need many iterative corrections |
| `max_depth` | Depth per base tree | Deeper → more interactions | Shallow (3-5) best with boosting | Best=5: captures age × education × employment interactions |
| `learning_rate` | Step size per update | Lower → needs more estimators | Low rate + many trees = best generalization | Best=0.1: balanced — not too aggressive, not too slow |
| `min_samples_leaf` | Min samples per leaf | Larger → more regularization | High → prevents overfitting | Best=5: prevents fitting to extreme income outliers |

---

## Task 3: GridSearchCV Results

| Model | Best Parameters | CV R² | Test R² | Combos |
|-------|----------------|-------|---------|--------|
| Decision Tree | max_depth=10, min_samples_split=2, min_samples_leaf=10 | 0.3404 | 0.3141 | 36 |
| Random Forest | n_estimators=100, max_depth=10, min_samples_leaf=5 | 0.3543 | 0.3252 | 18 |
| **Gradient Boosting** | **n_estimators=200, max_depth=5, learning_rate=0.1, min_samples_leaf=5** | **0.3629** | **0.3384** | **36** |

### Why These Parameters Won
- **max_depth=10 (DT, RF)**: depth 10 captures feature interactions without memorizing noise
- **max_depth=5 (GB)**: boosting builds many shallow trees; depth 5 × 200 stages = high effective complexity
- **min_samples_leaf=5-10**: prevents leaves from fitting to handful of extreme incomes
- **learning_rate=0.1**: moderate correction speed; 0.05 too slow, 0.2 overfits

---

## Task 4: Model Evaluation & Best Model Selection

### Full Comparison Table

| Model | Stage | R² | Adj R² | RMSE ($) | MAE ($) | MAPE (%) |
|-------|-------|-----|--------|----------|---------|----------|
| Linear Regression | D1 Baseline | 0.2358 | 0.2353 | 37,166 | 21,228 | 277.14 |
| Decision Tree | D1 Baseline | 0.1418 | 0.1412 | 39,388 | 20,496 | 266.37 |
| Random Forest | D1 Baseline | 0.2628 | 0.2623 | 36,505 | 19,099 | 220.37 |
| Decision Tree (Tuned) | D2 | 0.3141 | 0.3136 | 35,211 | 18,170 | 209.47 |
| Random Forest (Tuned) | D2 | 0.3252 | 0.3248 | 34,925 | 17,974 | 210.39 |
| **Gradient Boosting (Tuned)** | **D2** | **0.3384** | **0.3380** | **34,581** | **17,832** | **214.49** |

### D1 → D2 Improvement

| Model | Metric | D1 | D2 | Change |
|-------|--------|----|----|--------|
| Decision Tree | R² | 0.1418 | 0.3141 | +0.1723 (+121.5%) |
| Decision Tree | RMSE | $39,388 | $35,211 | -$4,177 (-10.6%) |
| Decision Tree | MAE | $20,496 | $18,170 | -$2,326 (-11.3%) |
| Random Forest | R² | 0.2628 | 0.3252 | +0.0624 (+23.7%) |
| Random Forest | RMSE | $36,505 | $34,925 | -$1,580 (-4.3%) |
| Random Forest | MAE | $19,099 | $17,974 | -$1,125 (-5.9%) |

### Best Model: Gradient Boosting (Tuned)
- R² = 0.3384 → explains 33.8% of income variation
- RMSE = $34,581
- MAE = $17,832
- Improvement over D1 best (RF): R² +28.8%, RMSE -5.3%, MAE -6.6%

### Why Each Model Performed This Way

| Model | Rank | Why |
|-------|------|-----|
| **Gradient Boosting (Tuned)** | #1 | Sequential error correction: each tree fixes previous mistakes; 200 stages × depth 5 = captures complex patterns |
| Random Forest (Tuned) | #2 | Parallel ensemble of 100 trees reduces variance; max_depth=10 prevents overfitting |
| Decision Tree (Tuned) | #3 | Single tree but depth capped at 10 → huge jump from D1 (+121% R²) |
| Random Forest (D1) | #4 | Good defaults but unlimited depth causes overfitting |
| Linear Regression | #5 | Cannot model non-linear interactions (age × education) |
| Decision Tree (D1) | #6 | Unlimited depth → memorized training data → worst generalization |

### Why Decision Tree Improved the Most (+121%)
- D1: unlimited depth → grew to hundreds of levels, memorized training noise
- D2: max_depth=10 + min_samples_leaf=10 → forced generalization
- Shows Decision Trees are extremely sensitive to depth control

### Why All Models Cap Around R² ≈ 0.34
- Missing key predictors: occupation, industry, hours worked, experience not in dataset
- Categorical granularity: age in 15 bins (not exact), education in 4 levels
- Within-group variance: same age group + education can earn $20K or $200K
- Extreme outliers: min=-$1.6M, max=$981K → impossible to predict from demographics

### Why Feature Transformations Did Not Help But Tuning Did
- Features are ordinal integers (1-44) → no wide ranges or extreme values to compress
- Tree-based models ignore feature scale → scaling has no effect on split decisions
- Hyperparameter tuning controls tree structure (depth, leaf size) → directly prevents overfitting
- The improvement came from **controlling model complexity**, not from feature preprocessing

### Evaluation Trade-offs
- **R² vs MAPE**: GB has best R² (0.338) but DT has best MAPE (209%) → GB explains more variance overall but relative % error is slightly higher
- **RMSE vs MAE gap**: RMSE ($34,581) ≈ 2× MAE ($17,832) → large errors on outlier predictions persist
- **CV R² vs Test R²**: small gap (0.363 vs 0.338 for GB) → model generalizes well, not just fitting training data
- **Complexity vs performance**: GB trains slower (200 sequential stages) but gains +1.3% R² over RF → worthwhile tradeoff
