# Deliverable 2 — Explanations

> **Format**: Bullet points and tables only — NO paragraphs.

---

## Task 1: Data Balancing — Why After-tax Income is Imbalanced

- Target variable: After-tax income (continuous, regression)
- Distribution: right-skewed (long tail of high earners)
- Outliers: extreme values at both ends (negatives from investment losses, high earners)
- IQR method flags significant % as outliers → unequal variance
- Majority of data clustered in $25K–$85K range
- Rare values at tails → model underpredicts extremes

### Transformation Trade-offs

| Transformation | Benefit | Risk / Limitation |
|---|---|---|
| Log Transform | Compresses right tail, reduces skewness | Requires shift for negatives/zeros; amplifies small differences near zero |
| Square-root Transform | Milder compression than log | Same shift requirement; less effective on heavy skew |
| Box-Cox Transform | Optimal power parameter (data-driven) | Requires strictly positive input; shift needed; lambda may overfit |
| Robust Scaling | Reduces influence of outliers via IQR-based scaling | Does not change distribution shape; skewness unchanged |

### Per-transformation Analysis

| Transformation | Reduces Skewness? | Stabilizes Variance? | More Predictable? |
|---|---|---|---|
| Log | Yes — strong compression of right tail | Yes — reduces spread | Yes — more symmetric distribution |
| Square-root | Partially — moderate compression | Partially | Moderate improvement |
| Box-Cox | Yes — optimally chosen power | Yes — data-driven variance stabilization | Best theoretical fit |
| Robust Scaling | No — shape unchanged | Yes — IQR-based normalization | Only helps scale-sensitive models |

### Impact on Prediction Errors
- MAE: more robust to outliers → lower for models that handle tails well
- RMSE: penalizes large errors → sensitive to outlier handling
- R²: overall fit quality → improved by reducing variance
- MAPE: percentage-based → affected by near-zero values
- See comparison table and bar chart in notebook

---

## Task 2: Hyperparameter Explanation

### Model 1: Decision Tree

| Hyperparameter | What It Controls | Model Complexity | Overfit vs Underfit | Why It Matters Here |
|---|---|---|---|---|
| max_depth | Maximum depth of the tree | Deeper → more complex | Deep → overfit (memorizes noise); Shallow → underfit | Income has non-linear relationships → needs depth, but unlimited depth memorizes training data |
| min_samples_split | Minimum samples required to split a node | Higher → simpler (fewer splits) | High → underfit; Low → overfit | Controls how granular the tree splits on income predictors |
| min_samples_leaf | Minimum samples in a leaf node | Higher → simpler (larger leaves) | High → underfit; Low → overfit | Prevents leaves with very few samples (rare income values at tails) |

### Model 2: Random Forest

| Hyperparameter | What It Controls | Model Complexity | Overfit vs Underfit | Why It Matters Here |
|---|---|---|---|---|
| n_estimators | Number of trees in the ensemble | More trees → more complex, diminishing returns | More trees → less overfit (averaging effect) | Large dataset (80K rows) → needs enough trees for stable predictions |
| max_depth | Maximum depth per tree | Deeper → more complex | Deep trees → overfit; Shallow → underfit | Income has non-linear relationships → needs depth, but too deep memorizes noise |
| min_samples_split | Minimum samples to split a node | Higher → simpler | High value → underfit; Low → overfit | Controls granularity of splits on income predictors |

### Model 3: SVR (Support Vector Regressor)

| Hyperparameter | What It Controls | Model Complexity | Overfit vs Underfit | Why It Matters Here |
|---|---|---|---|---|
| C | Regularization parameter | Higher C → more complex (less regularization) | High C → overfit (fits noise); Low C → underfit (too smooth) | Income data has outliers → C controls how much to penalize errors beyond epsilon tube |
| epsilon | Width of the epsilon-insensitive tube | Larger → ignores more errors | Large → underfit (too tolerant); Small → overfit (too sensitive) | Controls tolerance for prediction errors — wider tube = smoother predictions |
| kernel | Type of kernel function (rbf) | RBF → non-linear decision boundary | RBF captures complex patterns but can overfit | Income depends non-linearly on age, education, hours → RBF captures these interactions |

---

## Task 4: Interpretation

### Why Models Performed Differently
- Decision Tree: single tree → prone to overfitting, high variance, no ensemble averaging
- Random Forest: ensemble of trees → reduces variance via bagging, handles non-linearity well
- SVR: kernel-based → captures non-linear patterns, but slow on large datasets and sensitive to scaling
- Data characteristics influencing results:
  - High dimensionality (200+ features)
  - Mix of categorical codes and continuous values
  - Correlated features (person-level vs EF vs CF versions)
  - Skewed target with outliers
  - Large dataset (80K rows) → SVR may be slower than tree-based models

### Evaluation Trade-offs

| Consideration | Details |
|---|---|
| MAE vs RMSE | If RMSE >> MAE → model struggles with outlier predictions |
| R² interpretation | Higher = better fit, but can be inflated by correlated features |
| MAPE caveat | Unreliable for near-zero incomes → interpret with caution |
| Overfitting check | Compare CV score vs test score → large gap = overfitting |

### Deliverable 1 vs Deliverable 2

| Aspect | Deliverable 1 | Deliverable 2 |
|---|---|---|
| Data balancing | None | Applied target transformations |
| Hyperparameter tuning | Default parameters | GridSearchCV with 5-fold CV |
| Model comparison | Basic evaluation | Comprehensive with multiple metrics |
| Performance | *[Fill in D1 results]* | *[See notebook results]* |
| Key improvement | — | Transformation + tuning → better handling of skewed target |
