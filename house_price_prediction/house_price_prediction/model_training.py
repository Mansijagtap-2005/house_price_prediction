import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import pickle
import warnings
warnings.filterwarnings("ignore")

# ✅ Load dataset
data = pd.read_csv("housing_data.csv")
print("✅ Data loaded successfully!")

# ✅ Encode categorical columns
le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == "object":
        data[col] = le.fit_transform(data[col])

# ✅ Remove invalid / extreme values
data = data[data["price"] > 0]
data = data[data["area"] > 0]

# ✅ Feature engineering: create new meaningful features
data["rooms_total"] = data["bedrooms"] + data["bathrooms"]
data["price_per_sqft"] = data["price"] / (data["area"] + 1)

# ✅ Log-transform price for better regression stability
data["price_log"] = np.log1p(data["price"])

# ✅ Split data
X = data.drop(["price", "price_log"], axis=1)
y = data["price_log"]

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ✅ Train optimized XGBoost model
xgb_model = XGBRegressor(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.2,
    reg_alpha=0.5,
    gamma=0.1,
    random_state=42,
    objective="reg:squarederror"
)
xgb_model.fit(X_train, y_train)

# ✅ Train improved Random Forest
rf_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=12,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42
)
rf_model.fit(X_train, y_train)

# ✅ Combine (ensemble prediction)
xgb_pred = xgb_model.predict(X_test)
rf_pred = rf_model.predict(X_test)
final_pred = (0.7 * xgb_pred) + (0.3 * rf_pred)

# ✅ Convert back from log scale
y_true = np.expm1(y_test)
y_pred = np.expm1(final_pred)

# ✅ Evaluate
r2 = r2_score(y_true, y_pred)
print(f"🎯 Final Ensemble Accuracy (R²): {r2 * 100:.2f}%")

# ✅ Save models
with open("house_price_model.pkl", "wb") as f:
    pickle.dump((xgb_model, rf_model, scaler), f)

print("💾 Model saved successfully as 'house_price_model.pkl'")
