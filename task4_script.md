# Task 4: Model Evaluation & Best Model Selection

---

## 1. Compare model results -- which performed best?

| Model | Stage | R² | RMSE | MAE |
|-------|-------|----|------|-----|
| Decision Tree | Baseline | -0.0417 | 36,573 | 17,925 |
| Decision Tree | Transformed | -1.3159 | 54,532 | 31,854 |
| **Decision Tree** | **Tuned** | **0.4041** | **27,660** | **14,501** |
| Random Forest | Baseline | 0.2981 | 30,022 | 15,305 |
| Random Forest | Transformed | 0.0356 | 35,190 | 24,518 |
| **Random Forest** | **Tuned** | **0.4249** | **27,176** | **14,056** |
| SVR | Baseline | 0.0427 | 35,061 | 20,679 |
| SVR | Transformed | 0.0255 | 35,374 | 20,929 |
| **SVR** | **Tuned** | **0.3075** | **29,819** | **15,402** |

- **Random Forest (Tuned) is our best model.** It has the highest R² (0.4249), lowest RMSE ($27,176), and lowest MAE ($14,056) -- it wins on every metric.
- Decision Tree (Tuned) is a close second at R² 0.4041, only 0.02 behind RF.
- SVR (Tuned) ranks last at R² 0.3075, but still improved massively from its baseline of 0.04.
- Transformations hurt all models -- they distorted the ordinal structure of our features, making patterns harder to learn.
- GridSearchCV tuning was the biggest factor in improving every model.

---

## 2. Why did models perform better or worse?

### Random Forest -- best performer:
- As an ensemble of 200 trees, it averages out individual tree errors, which reduces variance and produces more stable predictions.
- It handles the mix of ordinal and continuous features naturally by splitting on thresholds, so features like age group and hours worked are both used effectively.
- The ensemble averaging makes it robust to outliers in our right-skewed income distribution -- no single extreme split dominates the final prediction.
- Tuned parameters (max_depth=15, min_samples_leaf=10) gave it enough complexity to capture real patterns without overfitting to noise.

### Decision Tree -- close second:
- As a single tree it has higher variance than RF because there is no ensemble to smooth out errors -- one bad split affects everything downstream.
- The tuned max_depth=10 prevented the overfitting that caused the baseline to get a negative R², but a single tree still cannot match the stability of 200 averaged trees.
- It captures the same non-linear feature interactions as RF, just less reliably.

### SVR -- last place:
- SVR fits a smooth continuous function across all features, but our dataset has ordinal variables with discrete jumps (age groups, education levels) -- the RBF kernel tries to interpolate smoothly between categories, which does not match how these features actually relate to income.
- SVR has O(n²) computational complexity, which limits how much training data it can use and how many hyperparameter combinations we can search, reducing its ability to find the optimal configuration.
- The epsilon-tube (0.5) ignores small residuals by design, meaning the model misses subtle income patterns that tree-based models capture through fine-grained splits.
- Despite ranking last, tuning improved its R² from 0.04 to 0.31 -- showing that hyperparameter optimization still made a big difference.

---

## 3. What data characteristics influence performance?

- **Right-skewed income distribution:** Most people earn moderate incomes but a few earn very high amounts. This causes all models to underpredict high earners and inflates RMSE, since RMSE penalizes large errors more heavily.
- **Outliers:** The gap between mean and median income shows outliers are present, which is why RMSE is always much higher than MAE across all models.
- **Mixed feature types:** Our 20 features include ordinal categories (age group, education) and continuous values (hours worked, weeks employed). Tree-based models split on these naturally, while SVR needs everything scaled and struggles with discrete jumps.
- **Key predictors:** Major source of income, total hours worked, and weeks employed are the strongest features -- these are direct income drivers that trees split on effectively.
- **MAPE unreliability:** MAPE exceeds 180% for all models because some individuals have near-zero incomes, causing percentage errors to explode. MAE is a more reliable error metric for this dataset.

---

## 4. Evaluation trade-offs

- **R²** tells us the proportion of variance explained, but not the dollar magnitude of errors -- R² of 0.42 still means ~$27K average error.
- **RMSE** penalizes large errors heavily, which is useful for catching big misses but gets inflated by our right-skewed income outliers.
- **MAE** gives the average dollar error and is more robust to outliers, but treats all errors equally -- a $1K miss and a $50K miss contribute the same.
- **MAPE** is unreliable here because near-zero incomes cause division by tiny numbers, producing inflated percentages.
- **RMSE-MAE gap** reveals outlier sensitivity: Random Forest has the smallest gap ($13,120), meaning it is the most consistent across income ranges. SVR has the largest gap ($14,417), meaning it makes some very large errors on extreme incomes.

---

## 5. Deliverable 1 vs Deliverable 2

### Key differences:
- D1 used default hyperparameters; D2 used GridSearchCV with 5-fold cross-validation.
- D1 used one-hot encoding (100+ columns); D2 used 20 numeric features directly.
- D1 included Linear Regression; D2 replaced it with a tuned SVR.

### Performance comparison:

| Model | D1 R² | D2 R² | D1 RMSE | D2 RMSE |
|-------|-------|-------|---------|---------|
| Decision Tree | 0.1418 | 0.4041 | 39,388 | 27,660 |
| Random Forest | 0.2628 | 0.4249 | 36,505 | 27,176 |
| SVR | -0.0182 | 0.3075 | 42,902 | 29,819 |

- All three models improved dramatically -- RMSE dropped by ~30% ($10K-$13K reduction per prediction).
- Decision Tree improved the most in absolute R² (0.14 to 0.40) because D1 had no depth limit causing severe overfitting, while D2 tuned max_depth=10 to regularize it.
- SVR had the largest relative improvement (-0.02 to 0.31) because D1's default C=1.0 was far too weak; D2 tuned C=1000 for much stronger learning.
- Hyperparameter tuning was the single biggest driver of improvement across all models.

---

## 6. Final Verdict

| Rank | Model | R² | RMSE | MAE |
|------|-------|----|------|-----|
| 1 | **Random Forest** | **0.4249** | **$27,176** | **$14,056** |
| 2 | Decision Tree | 0.4041 | $27,660 | $14,501 |
| 3 | SVR | 0.3075 | $29,819 | $15,402 |

- **Random Forest (Tuned) is the best model** -- it wins every metric, is most robust to outliers, and its 200-tree ensemble provides the most reliable predictions for unseen data.
