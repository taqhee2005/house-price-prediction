from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parent
DATA = BASE / "dataset" / "house_prices.csv"
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
target = "price_lakh"

X = df.drop(columns=[target])
y = df[target]

numeric_features = ["area", "bedrooms", "bathrooms", "stories", "parking"]
categorical_features = ["location", "furnishing"]

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_features),
    ("cat", categorical_pipe, categorical_features)
])

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=300, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

results = {}
fitted = {}

for name, estimator in models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator)
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    results[name] = {
        "MAE_lakh": round(mean_absolute_error(y_test, pred), 4),
        "RMSE_lakh": round(mean_squared_error(y_test, pred) ** 0.5, 4),
        "R2": round(r2_score(y_test, pred), 4)
    }
    fitted[name] = pipe

best_name = max(results, key=lambda n: results[n]["R2"])
best_model = fitted[best_name]

joblib.dump(best_model, MODEL_DIR / "house_price_model.pkl")

with open(MODEL_DIR / "model_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "best_model": best_name,
        "results": results
    }, f, indent=2)

print("Model comparison:")
for name, metrics in results.items():
    print(f"{name}: {metrics}")

print(f"\nSaved best model: {best_name}")
print(f"Model file: {MODEL_DIR / 'house_price_model.pkl'}")
