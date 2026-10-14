import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Reproducibility
np.random.seed(42)

# Number of synthetic records
n_samples = 300

# Features
monthly_savings_consistency = np.random.uniform(0, 1, n_samples)
rent_and_bill_payment_history = np.random.uniform(0, 1, n_samples)
spending_to_income_ratio = np.random.uniform(0, 1, n_samples)

# Create target with a realistic relationship
score = (
    0.45 * monthly_savings_consistency +
    0.40 * rent_and_bill_payment_history -
    0.50 * spending_to_income_ratio +
    np.random.normal(0, 0.15, n_samples)
)

loan_repayment_success = (score > 0.25).astype(int)

# Build dataframe
df = pd.DataFrame({
    "monthly_savings_consistency": monthly_savings_consistency,
    "rent_and_bill_payment_history": rent_and_bill_payment_history,
    "spending_to_income_ratio": spending_to_income_ratio,
    "loan_repayment_success": loan_repayment_success
})

# Save synthetic dataset
df.to_excel("fintech.xlsx", index=False)

# Define X and y
X = df[[
    "monthly_savings_consistency",
    "rent_and_bill_payment_history",
    "spending_to_income_ratio"
]]
y = df["loan_repayment_success"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# Evaluation
print("TRAIN CLASSIFICATION REPORT")
print(classification_report(y_train, train_pred))

print("TEST CLASSIFICATION REPORT")
print(classification_report(y_test, test_pred))

# Save model
joblib.dump(model, "fintech.joblib")

print("Saved fintech.xlsx and fintech.joblib successfully.")
