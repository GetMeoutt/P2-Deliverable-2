# Deliverable 2 - Explanations

---

## Task 1: Data Balancing (Regression)

### Target Variable: After-tax Income
- Continuous numeric target → **Regression problem**
- Right-skewed distribution (positive skewness)
- Contains negative values (tax adjustments) and zeros
- Significant outliers on the high end (extreme earners)

### Why This Is Imbalance (Regression Context)
- Most incomes clustered in $20K–$60K range
- Few extreme values ($200K+) pull the mean away from the median
- Outliers inflate RMSE disproportionately
- Models overfit to the dense middle range, underperform on tails

### Transformations Applied

| # | Technique | Applied To | Purpose |
|---|-----------|-----------|---------|
| 1 | Log Transform | Target (y) | Compress right tail, reduce skewness |
| 2 | Square-root Transform | Target (y) | Moderate compression, less aggressive than log |
| 3 | Box-Cox Transform | Target (y) | Optimal power transform (data-driven lambda) |
| 4 | Robust Scaling | Features (X) | Scale features using median/IQR, reduce outlier influence |

### How Each Transformation Works

**Log Transform**
- Formula: y' = log(y + shift)
- Inverse: y = exp(y') - shift
- Effect: strongly compresses large values, spreads small values

**Square-root Transform**
- Formula: y' = sqrt(y + shift)
- Inverse: y = (y')² - shift
- Effect: milder compression than log

**Box-Cox Transform**
- Formula: y' = (y^lambda - 1) / lambda (lambda estimated from data)
- Inverse: scipy inv_boxcox function
- Effect: finds optimal power transformation automatically

**Robust Scaling**
- Formula: x' = (x - median) / IQR
- Applied to features, not target
- Effect: centers features around median, scales by IQR instead of std

### Trade-offs

| Technique | Benefit | Risk |
|-----------|---------|------|
| Log Transform | Reduces skewness significantly | Distorts original relationships; requires shift for negatives |
| Sqrt Transform | Moderate skew reduction | Less effective on heavy-tailed distributions |
| Box-Cox | Optimal lambda, best normality | Sensitive to outliers; requires strictly positive values |
| Robust Scaling | Outlier-resistant feature scaling | No effect on tree-based models (split-invariant to scaling) |

### Interpretation Criteria
- **Reduces skewness** → distribution closer to symmetric (skewness → 0)
- **Stabilizes variance** → spread more uniform across the range
- **More predictable patterns** → model captures relationships better (higher R²)

### Evaluation Metrics Used

| Metric | What It Measures | Sensitivity |
|--------|-----------------|-------------|
| MAE | Average absolute error in $ | Robust to outliers |
| RMSE | Root mean squared error in $ | Penalizes large errors heavily |
| R² | Proportion of variance explained | Overall fit quality |
| Adjusted R² | R² penalized for # of features | Prevents overfitting to feature count |
| MAPE | % error relative to actual value | Interpretable scale; undefined at y=0 |

---

## Task 2: Hyperparameter Explanation

### Model 1: Decision Tree Regressor

| Hyperparameter | What It Controls | Impact on Complexity | Overfitting vs Underfitting | Why It Matters for This Dataset |
|----------------|-----------------|---------------------|---------------------------|-------------------------------|
| `max_depth` | Maximum tree depth | Higher depth → more complex | Deep trees overfit; shallow trees underfit | 81K rows with 11 features → needs enough depth to capture interactions but not memorize noise |
| `min_samples_split` | Minimum samples to split a node | Lower → more splits → more complex | Low values → overfit to small groups | Income has many subgroups (age × education); too few samples per split = unreliable patterns |
| `min_samples_leaf` | Minimum samples in a leaf node | Lower → smaller leaves → more complex | Low values → overfit; high values → oversmooth | Prevents leaf nodes with 1-2 extreme income values from dominating predictions |

### Model 2: Random Forest Regressor

| Hyperparameter | What It Controls | Impact on Complexity | Overfitting vs Underfitting | Why It Matters for This Dataset |
|----------------|-----------------|---------------------|---------------------------|-------------------------------|
| `n_estimators` | Number of trees in the forest | More trees → more computation, marginal accuracy gain | More trees reduce variance (less overfit) | 81K rows benefit from ensemble averaging; diminishing returns after ~100-200 trees |
| `max_depth` | Maximum depth per tree | Deeper trees capture more patterns | Deep + many trees can still overfit | Limits how specific each tree gets; balances bias-variance for noisy income data |
| `min_samples_leaf` | Minimum samples per leaf | Larger → simpler trees | High values prevent overfitting to outlier incomes | Income outliers ($500K+) should not dominate individual leaf predictions |

### Model 3: Gradient Boosting Regressor

| Hyperparameter | What It Controls | Impact on Complexity | Overfitting vs Underfitting | Why It Matters for This Dataset |
|----------------|-----------------|---------------------|---------------------------|-------------------------------|
| `n_estimators` | Number of boosting stages | More stages → more complex | Too many → overfit; too few → underfit | Sequential correction of errors; income prediction errors need iterative refinement |
| `max_depth` | Depth of each base tree | Deeper → captures more interactions | Shallow trees (3-5) work best with boosting | Interaction between age, education, employment → needs moderate depth |
| `learning_rate` | Step size for each update | Lower → needs more estimators | Low rate + many trees = best generalization; high rate = overfit fast | Controls how aggressively model corrects errors; lower = more stable for noisy income data |
| `min_samples_leaf` | Minimum samples per leaf | Larger → more regularization | High values prevent overfitting | Protects against fitting to extreme income values in individual boosting stages |

---

## Task 4: Model Evaluation & Best Model Selection

### Performance Comparison
- See summary table and bar charts in notebook (comparison_df)
- All models evaluated on the **original test set** (untransformed)

### Why Models Performed Better or Worse

| Model | Performance Level | Key Reason |
|-------|-----------------|------------|
| Gradient Boosting (Tuned) | Best | Sequential error correction + tuned hyperparameters capture complex income patterns |
| Random Forest (Tuned) | Strong | Ensemble averaging reduces variance; tuning optimizes tree structure |
| Decision Tree (Tuned) | Moderate | Single tree limited by greedy splits; tuning prevents overfitting |
| Linear Regression | Moderate | Assumes linear relationships; cannot capture age × education interactions |
| Random Forest (Default) | Good baseline | Reasonable defaults but not optimized for this specific data |
| Decision Tree (Default) | Weak | Unlimited depth → overfits training data |

### Data Characteristics Influencing Results
- **All features are categorical/ordinal** → tree-based models naturally handle this
- **Non-linear relationships** → age group and education interact; linear models miss this
- **Right-skewed target** → outlier incomes inflate RMSE for all models
- **Missing key predictors** → occupation, industry, hours worked are unavailable → caps all models' R²

### D1 vs D2 Comparison

| Aspect | D1 (Baseline) | D2 (Tuned) |
|--------|---------------|------------|
| Hyperparameters | Default | GridSearchCV-optimized (5-fold CV) |
| Data Balancing | None | Target transformations tested |
| Best Model | Random Forest (default) | Gradient Boosting or Random Forest (tuned) |
| R² | ~0.26 | Improved (see notebook) |
| RMSE | ~$36,500 | Reduced (see notebook) |
| MAE | ~$19,100 | Reduced (see notebook) |

### Key Improvements in D2
- Hyperparameter tuning → reduced overfitting, improved generalization
- Target transformations → tested skewness reduction impact
- Gradient Boosting → new model that sequentially corrects errors
- GridSearchCV with 5-fold CV → robust parameter selection, avoids lucky splits

### Evaluation Trade-offs
- **R² vs MAE**: high R² model may still have large errors on outliers
- **RMSE vs MAE**: when RMSE >> MAE → outlier predictions are poor
- **Train R² vs Test R²**: large gap = overfitting (tuning reduces this gap)
- **Complexity vs Performance**: Gradient Boosting is slower to train but more accurate
