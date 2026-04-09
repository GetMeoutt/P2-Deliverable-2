# Task 4: Model Evaluation & Best Model Selection

---

## 1. Compare model results -- which performed best?

| Model | Stage | R² | Adj R² | RMSE | MAE | MAPE |
|-------|-------|----|--------|------|-----|------|
| Decision Tree | Baseline | -0.0417 | -0.0428 | 36,573 | 17,925 | 227.53 |
| Decision Tree | Transformed | -1.3159 | -1.3183 | 54,532 | 31,854 | 357.86 |
| **Decision Tree** | **Tuned** | **0.4041** | **0.4035** | **27,660** | **14,501** | **187.99** |
| Random Forest | Baseline | 0.2981 | 0.2973 | 30,022 | 15,305 | 209.73 |
| Random Forest | Transformed | 0.0356 | 0.0346 | 35,190 | 24,518 | 379.65 |
| **Random Forest** | **Tuned** | **0.4249** | **0.4243** | **27,176** | **14,056** | **180.24** |
| SVR | Baseline | 0.0427 | 0.0417 | 35,061 | 20,679 | 312.95 |
| SVR | Transformed | 0.0255 | 0.0245 | 35,374 | 20,929 | 314.87 |
| **SVR** | **Tuned** | **0.3075** | **0.3068** | **29,819** | **15,402** | **186.92** |

- Random Forest (Tuned) is the best model because it has the highest R² of 0.4249, meaning it explains about 42% of the variance in after-tax income.
- It also has the lowest RMSE ($27,176) and lowest MAE ($14,056), which means its predictions are closest to the actual values on average.
- Decision Tree (Tuned) comes in second with R² of 0.4041, very close to RF but slightly less accurate.
- SVR (Tuned) ranks last with R² of 0.3075 because it was trained on only a 5,000-row subsample due to its O(n²) computational cost.
- Transformations actually hurt all models -- Decision Tree went from R² -0.04 to -1.32, showing that skewness-reducing transforms distorted the ordinal structure of the features.
- GridSearchCV tuning was the single biggest factor in improving performance across all three models.

---

## 2. Why did models perform better or worse for this dataset?

### Random Forest -- why it performed best:
- It uses an ensemble of 200 decision trees, which averages out individual tree errors and reduces overall prediction variance.
- It naturally captures non-linear interactions between features, such as how the combination of age group, education level, and hours worked together influence income.
- The ensemble averaging makes it robust to outliers in the right-skewed income distribution because no single extreme split dominates the final prediction.
- The tuned parameters (max_depth=15, min_samples_leaf=10) gave it enough complexity to learn real patterns without memorizing noise.

### Decision Tree -- why it performed close but worse:
- As a single tree, it has higher variance than RF because there is no ensemble to smooth out prediction errors.
- The tuned max_depth=10 provided good regularization and prevented the overfitting that plagued the baseline (which had no depth limit and got R² of -0.04).
- It is only 0.02 R² behind RF, but its predictions are less stable across different cross-validation folds.

### SVR -- why it ranked last:
- It was trained on only 5,000 rows (subsampled from the full training set) because SVR has O(n²) time complexity, meaning it saw less training data than DT and RF.
- The RBF kernel with gamma='scale' has limited expressiveness when working with 20 mixed features that include both ordinal categories and continuous values.
- The epsilon-tube width of 0.5 means the model ignores residuals smaller than 0.5, which causes it to miss subtle income patterns.
- Despite ranking last, it still improved massively from a baseline R² of 0.04 to a tuned R² of 0.31.

---

## 3. What data characteristics are influencing model performance?

- The target variable (after-tax income) has a right-skewed distribution, meaning most people earn moderate incomes but a few earn very high amounts -- this causes all models to underpredict high earners and inflates RMSE.
- The gap between mean and median income shows the presence of outliers, which is why RMSE (sensitive to large errors) is always much higher than MAE (robust to outliers).
- The dataset has 20 numeric features that are a mix of ordinal categories (age group, education level) and continuous values (hours worked, weeks employed) -- tree-based models handle this mix naturally, while SVR treats everything as continuous and needs scaling.
- The top predictive features are "Major source of income," "Total hours worked," and "Number of weeks employed" -- these are labor and income-source features that tree models split on very effectively.
- Several features are low-cardinality ordinal variables (binary flags, age groups with few categories) -- trees partition these efficiently, but SVR struggles because it tries to fit a smooth function over discrete jumps.
- MAPE values are above 180% for all models because some individuals have near-zero incomes, causing the percentage error to explode -- this makes MAPE unreliable for this dataset and MAE a better error metric.

---

## 4. Evaluation trade-offs

- R² and Adjusted R² tell us how much variance the model explains, but they do not tell us the actual dollar amount of prediction errors -- a model with R² of 0.42 still has $27K average error.
- RMSE penalizes large errors heavily (squared), which is useful for catching big misses, but it gets inflated by the right-skewed income outliers in our dataset -- making models look worse than they are for typical predictions.
- MAE gives the average dollar error and is more robust to extreme outliers, but it treats a $1,000 error the same as a $50,000 error -- so it can hide poor performance on high earners.
- MAPE should give a scale-independent percentage error, but it is unreliable here because near-zero incomes cause division by very small numbers, producing 180%+ values even for the best model.
- The RMSE-MAE gap reveals how much a model is affected by outliers:
  - Random Forest has the smallest gap ($13,120), meaning it is the most consistent across all income ranges.
  - SVR has the largest gap ($14,417), meaning it makes some very large errors on extreme incomes.
  - Decision Tree falls in between ($13,159).
- Adjusted R² is preferred over R² because it penalizes adding unnecessary features -- with 20 features, this prevents us from being fooled by overfitting.

---

## 5. Model performance differences between Deliverable 1 and Deliverable 2

### What was different between D1 and D2:
- D1 used default hyperparameters with no tuning, while D2 used GridSearchCV with 5-fold cross-validation to find optimal parameters.
- D1 used one-hot encoding which expanded categorical features into 100+ columns, while D2 treated all 20 features as numeric directly.
- D1 included Linear Regression as a model, while D2 replaced it with a properly tuned SVR.

### Performance comparison:

| Model | D1 R² | D2 R² (Tuned) | R² Change | D1 RMSE | D2 RMSE | RMSE Change |
|-------|-------|---------------|-----------|---------|---------|-------------|
| Decision Tree | 0.1418 | 0.4041 | +185% | 39,388 | 27,660 | -30% |
| Random Forest | 0.2628 | 0.4249 | +62% | 36,505 | 27,176 | -26% |
| SVR | -0.0182 | 0.3075 | +1,789% | 42,902 | 29,819 | -30% |

- Decision Tree improved the most in absolute R² terms (from 0.14 to 0.40) because D1 had no depth limit causing severe overfitting, while D2 tuned max_depth=10 and min_samples_leaf=20 to regularize it.
- Random Forest improved from 0.26 to 0.42 because D2 increased trees from 100 to 200 and tuned max_depth=15 with min_samples_leaf=10 for better bias-variance balance.
- SVR had the largest relative improvement (from -0.02 to 0.31) because D1 used default C=1.0 which was far too weak, while D2 tuned C=1000 giving the model much stronger regularization power.
- RMSE dropped by about 30% for all three models, meaning the average prediction error decreased by roughly $10,000-$13,000 per prediction.
- Hyperparameter tuning was the single biggest driver of improvement -- the same algorithms with better parameters performed dramatically better.
- Using numeric features directly (instead of one-hot) reduced noise from sparse dummy columns and gave the models cleaner signal to learn from.

---

## 6. Final Verdict

| Rank | Model | R² | RMSE | MAE | Best Parameters |
|------|-------|----|------|-----|-----------------|
| 1 | **Random Forest** | **0.4249** | **$27,176** | **$14,056** | n_estimators=200, max_depth=15, min_samples_leaf=10 |
| 2 | Decision Tree | 0.4041 | $27,660 | $14,501 | max_depth=10, min_samples_split=50, min_samples_leaf=20 |
| 3 | SVR | 0.3075 | $29,819 | $15,402 | C=1000, epsilon=0.5, gamma=scale |

- Random Forest (Tuned) is the best model because it wins every single evaluation metric.
- It is the most robust to income outliers as shown by the smallest RMSE-MAE gap.
- Its ensemble of 200 trees reduces prediction variance, making it the most reliable model for unseen data.
- It has the best generalization performance with the smallest gap between training R² and test R².
