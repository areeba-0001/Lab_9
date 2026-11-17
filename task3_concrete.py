import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

concrete = pd.read_csv("concrete_strength.csv").drop_duplicates()

target = "Concrete compressive strength"
X = concrete.drop(columns=target)
y = concrete[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

folds = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = "neg_root_mean_squared_error"

linear_model = LinearRegression()

linear_scores = -cross_val_score(
    linear_model, X_train, y_train, cv=folds, scoring=scoring
)

print("Linear Mean CV RMSE:", linear_scores.mean())

ridge_model = Pipeline([
    ("poly", PolynomialFeatures(include_bias=False)),
    ("scale", StandardScaler()),
    ("ridge", Ridge())
])

param_grid = {
    "poly__degree": [1, 2, 3],
    "ridge__alpha": [0.01, 0.1, 1, 10, 100]
}

search = GridSearchCV(
    ridge_model, param_grid, cv=folds,
    scoring=scoring, refit=True
)

search.fit(X_train, y_train)

print("Best Parameters:", search.best_params_)
print("Best Mean CV RMSE:", -search.best_score_)

results = pd.DataFrame(search.cv_results_)
top_three = results.sort_values("rank_test_score").head(3)

print(top_three[[
    "param_poly__degree",
    "param_ridge__alpha",
    "rank_test_score",
    "mean_test_score"
]])

test_predictions = search.predict(X_test)

test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
test_mae = mean_absolute_error(y_test, test_predictions)

print("Test RMSE:", test_rmse)
print("Test MAE:", test_mae)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, test_predictions)

minimum = min(y_test.min(), test_predictions.min())
maximum = max(y_test.max(), test_predictions.max())

plt.plot([minimum, maximum], [minimum, maximum],
         linestyle="--", color="red")

plt.xlabel("Actual Concrete Strength (MPa)")
plt.ylabel("Predicted Concrete Strength (MPa)")
plt.title("Actual vs Predicted Concrete Strength")
plt.grid()
plt.savefig("task3_concrete.png", dpi=300, bbox_inches="tight")
plt.show()
