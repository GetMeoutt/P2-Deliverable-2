# Deliverable 2 — Explanations

---

## Task 1: Data Balancing (Handling Outliers & Unequal Variance)

### Identifying Outliers Across All 20 Independent Variables

| Observation | Evidence |
|---|---|
| Outliers detected via IQR method on all 20 features | Boxplots show points beyond Q1 - 1.5×IQR and Q3 + 1.5×IQR |
| Many features have high outlier % | e.g., Total hours worked (41.5%), Economic family type (31.6%), Census family composition (25.5%) |
| Features with IQR = 0 flag all non-modal values as outliers | e.g., Flag - Person completed high school, Full-time/part-time student |
| 15 of 20 features are categorical/ordinal (≤10 unique values) | Outlier detection via IQR is less meaningful for these — "outliers" are just rare categories |

### Why This is Imbalanced

| Factor | Evidence |
|---|---|
| Target (After-tax Income) is right-skewed | Histogram shows long right tail; median (~$40K) much lower than mean (~$47K) |
| Feature distributions are skewed | Skewness table shows values up to 2.1 (Flag - Person completed high school) and -2.5 (Full-time/part-time student) |
| Rare values dominate some features | Boxplots show many features with 10-40% of values flagged as outliers |

→ Models biased toward the dense center of each feature's distribution, underperform on rare value combinations

---

### 4 Transformations Considered

| Transformation | What It Does | Best For |
|---|---|---|
| **Log** | Compresses large values, expands small values | Right-skewed features with large range |
| **Square-root** | Milder compression than log | Moderate skew, count-like data |
| **Box-Cox** | Power transformation, finds optimal lambda | Any skew direction, continuous data |
| **Robust Scaling** | Centers on median, scales by IQR | Features with many outliers, preserves shape |

### Selection Criteria
- For each feature → applied transformations and selected the one that **minimizes |skewness|**
- **Continuous features** (>10 unique values) → all 4 transformations considered
- **Categorical/ordinal features** (≤10 unique values) → Box-Cox excluded (distorts ordinal structure); Log, Square-root, Robust Scaling considered
- Every feature receives a transformation — none left untouched

---

### Trade-offs

| Technique | Advantage | Risk |
|---|---|---|
| Log Transform | Reduces right skew effectively | Requires shift for zero/negative values; distorts spacing |
| Square-root | Gentler than log, intuitive | Less effective on heavy skew |
| Box-Cox | Optimal power parameter auto-selected | Requires strictly positive input (shifted); distorts categorical/ordinal features |
| Robust Scaling | Handles outliers without removing them | Does not change distribution shape (skew unchanged) |

---

### What the Transformations Achieve

| Goal | How to Verify (from graphs) |
|---|---|
| **Reduces skewness** | Before/after skew values shown on each of the 20 feature plots |
| **Stabilizes variance** | Transformed distributions are more symmetric, less spread in tails |
| **Makes patterns more predictable** | Before vs After model metrics (R², RMSE, MAE) comparison chart |

---

### Evaluation Metrics Used

| Metric | Why It Matters |
|---|---|
| **MAE** | Robust to outliers — gives average absolute error |
| **RMSE** | Penalizes large errors more — sensitive to outlier predictions |
| **R²** | Proportion of variance explained — overall model fit |
| **Adjusted R²** | R² corrected for number of features |
| **MAPE** | Percentage error — intuitive interpretation |

### How Imbalance Impacts Predictions
- **Before transformation**: Models biased toward predicting median income range
- **RMSE >> MAE** indicates large errors on extreme values (high-income outliers)

### Why Transformations Decreased Performance

| Factor | Explanation |
|---|---|
| **Most features are categorical/ordinal** | 15 of 20 features have ≤10 unique values (flags, codes, groups) — not truly continuous |
| **Tree models split on thresholds** | DT/RF find splits like "Economic family type ≤ 23" — transformations warp spacing between categories, making splits less clean |
| **RF dropped from 0.30 → 0.04 R²** | Every tree in the ensemble got worse splits → ensemble averaging couldn't compensate |
| **DT went from -0.04 → -1.31 R²** | Single tree relies entirely on clean thresholds — warped values caused catastrophic splits |
| **SVR barely changed** | Already uses StandardScaler — additional transforms on top added noise, not signal |
| **Skewness ≠ model performance** | Lower skewness helps models that assume normality (linear regression); tree models are inherently robust to skew |

### Key Takeaway

| Model Type | Benefits from Skew Reduction? | Why |
|---|---|---|
| **Decision Tree** | No | Splits on rank order — only cares about thresholds, not distribution shape |
| **Random Forest** | No | Same as DT; ensemble averaging already handles variance |
| **SVR** | Minimal | StandardScaler already normalizes feature ranges; further transforms redundant |

→ Transformations are **not universally beneficial** — they can hurt when applied to categorical/ordinal data or models that don't assume normality

---

## Task 2: Hyperparameter Explanation

### Decision Tree Regressor

| Hyperparameter | What It Controls | Impact on Complexity | Overfitting vs Underfitting | Why It Matters for This Dataset |
|---|---|---|---|---|
| **max_depth** | Maximum tree depth | Higher → more complex | High depth → overfits; Low depth → underfits | 20 categorical/ordinal features — deep trees memorize specific feature combinations |
| **min_samples_split** | Minimum samples to split a node | Lower → more complex | Low value → overfits to small groups; High value → underfits | Prevents splits on tiny subgroups of income categories |
| **min_samples_leaf** | Minimum samples in a leaf node | Lower → more complex | Low value → overfits; High value → underfits | With 77K+ rows, small leaves can memorize noise |

**Validation curve insight**:
- Train R² ≈ 1.0 at high depth / low min_samples → clear overfitting
- CV R² peaks at moderate depth → optimal complexity tradeoff

---

### Random Forest Regressor

| Hyperparameter | What It Controls | Impact on Complexity | Overfitting vs Underfitting | Why It Matters for This Dataset |
|---|---|---|---|---|
| **n_estimators** | Number of trees in ensemble | More trees → diminishing returns, not more complex per tree | More trees → reduces variance (less overfitting); Few trees → high variance | Stabilizes noisy income predictions; diminishing returns after ~100 trees |
| **max_depth** | Maximum depth per tree | Higher → each tree more complex | High depth → individual trees overfit (but ensemble averaging helps); Low depth → underfits | Controls how deeply each tree can model income patterns |
| **min_samples_leaf** | Minimum samples per leaf | Lower → more complex | Low → overfits; High → underfits | Prevents individual trees from creating leaves for single income outliers |

**Validation curve insight**:
- n_estimators: CV score improves then plateaus — more trees = more stable, not more complex
- max_depth: Train score stays high, CV peaks mid-range → ensemble reduces but doesn't eliminate overfitting

---

### SVR (Support Vector Regressor)

| Hyperparameter | What It Controls | Impact on Complexity | Overfitting vs Underfitting | Why It Matters for This Dataset |
|---|---|---|---|---|
| **C** | Regularization strength (inverse) | Higher C → less regularization → more complex | High C → overfits (fits every point); Low C → underfits (too smooth) | Income data has outliers — need balanced C to avoid fitting extreme values |
| **epsilon** | Width of insensitive tube | Smaller → more support vectors → more complex | Small epsilon → overfits; Large epsilon → underfits (ignores too much) | Controls how much prediction error is tolerated before penalizing |
| **gamma** | RBF kernel width | Higher → tighter kernels → more complex | High gamma → overfits (local patterns only); Low gamma → underfits (too smooth) | With 20 features, gamma controls how many nearby points influence each prediction |

**Validation curve insight**:
- C: Very high C → train R² high but CV drops → overfitting
- gamma: High gamma → model fits training noise, CV collapses
- Sweet spot exists at moderate C and gamma values
