import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib
import os

np.random.seed(42)

# -----------------------------
# Generate realistic dataset
# -----------------------------
N = 2000

data = pd.DataFrame({
    "income": np.random.randint(20000, 150000, N),
    "credit_score": np.random.randint(300, 850, N),
    "loan_amount": np.random.randint(5000, 60000, N),
    "age": np.random.randint(21, 65, N),
    "employment_years": np.random.randint(0, 30, N)
})

# Approval logic (hidden pattern)
data["approved"] = (
    (data["credit_score"] > 650) &
    (data["income"] > 40000) &
    (data["loan_amount"] < data["income"] * 0.45) &
    (data["employment_years"] > 1)
).astype(int)

# -----------------------------
# Train model
# -----------------------------
X = data.drop("approved", axis=1)
y = data["approved"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    random_state=42
)

model.fit(X_train, y_train)

print("✅ Accuracy:", model.score(X_test, y_test))

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")

print("✅ Model saved!")