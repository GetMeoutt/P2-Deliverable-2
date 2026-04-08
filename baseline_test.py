import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

df = pd.read_csv('CIS2022_PUMF_CLEANED.csv')
target = 'After-tax income'
df = df.dropna(subset=[target])

selected_features = [
    'Major source of income',
    "Person's age group as of December 31 of refyear",
    'Total usual hours worked at all jobs during refyear',
    'Yearly summary of time worked during the refyear',
    'Flag - Person was self-employed during refyear',
    'Flag - Person was a paid employee during refyear',
    'Number of weeks employed during refyear',
    'Flag - Person was employed during the reference year',
    'Average usual hours worked per week at all jobs during refyear',
    'Highest level of education of person, 2nd grouping',
    'Annual labour force status',
    'Flag - Attended school, college, CEGEP or university during refyear',
    'Age group of youngest person in economic family',
    'Full-time or part-time student during refyear',
    'Economic family type',
    'Number of weeks not in the labour force during refyear',
    'Age group of oldest person in economic family',
    'Marital status',
    'Flag - Person completed high school',
    'Census family composition',
]

X = df[selected_features].copy()
y = df[target].copy()

# Clean special codes + impute
for col, codes in {
    'Average usual hours worked per week at all jobs during refyear': [999.6],
    'Total usual hours worked at all jobs during refyear': [9996],
    'Number of weeks employed during refyear': [96, 99],
    'Number of weeks not in the labour force during refyear': [96, 99],
}.items():
    if col in X.columns:
        X[col] = X[col].replace(codes, np.nan)

for col in X.columns:
    if X[col].isnull().any():
        X[col] = X[col].fillna(X[col].median())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Baseline R2 Scores:")
for name, model in [('Linear Regression', LinearRegression()),
                     ('Decision Tree', DecisionTreeRegressor(random_state=42)),
                     ('Random Forest', RandomForestRegressor(random_state=42))]:
    model.fit(X_train, y_train)
    print(f"  {name:25s} R2={r2_score(y_test, model.predict(X_test)):.4f}")
