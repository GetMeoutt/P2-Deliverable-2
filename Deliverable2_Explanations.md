# Deliverable 2 — Explanations

---

## Task 1: Data Balancing (Handling Outliers & Unequal Variance)

### Why This is a Regression Imbalance Problem

| Observation | Evidence |
|---|---|
| Target (After-tax Income) is right-skewed | Histogram shows long right tail |
| Extreme outliers exist | Boxplot shows values far beyond Q3 + 1.5×IQR |
| Unequal variance | Boxplots show wide spread with many points beyond whiskers |
| Most values clustered in low-mid range | Median (~$40K) much lower than mean (~$47K) |

→ Models overfit to the dense low-income region, underperform on high-income predictions

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
- Different features may use different transformations

---

### Trade-offs

| Technique | Advantage | Risk |
|---|---|---|
| Log Transform | Reduces right skew effectively | Requires shift for zero/negative values; distorts spacing |
| Square-root | Gentler than log, intuitive | Less effective on heavy skew |
| Box-Cox | Optimal power parameter auto-selected | Requires strictly positive input (shifted); can overfit to training distribution |
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
| **max_depth** | Maximum tree depth | Higher → more complex | High depth → overfits; Low depth → underfits | Income data has non-linear patterns; need enough depth to capture them without memorizing noise |
| **min_samples_split** | Minimum samples to split a node | Lower → more complex | Low value → overfits to small groups; High value → underfits | Many features have discrete/categorical values — prevents splits on tiny subgroups |
| **min_samples_leaf** | Minimum samples in a leaf node | Lower → more complex | Low value → overfits; High value → underfits | With 45K+ rows, small leaves can memorize individual income patterns |

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
| **gamma** | RBF kernel width | Higher → tighter kernels → more complex | High gamma → overfits (local patterns only); Low gamma → underfits (too smooth) | With 20 features, gamma controls how many features influence each prediction |

**Validation curve insight**:
- C: Very high C → train R² high but CV drops → overfitting
- gamma: High gamma → model fits training noise, CV collapses
- Sweet spot exists at moderate C and gamma values
