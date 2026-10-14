from flask import Flask, render_template, request
import pandas as pd
from joblib import load

app = Flask(__name__)

model = load("fintech.joblib")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET"])
def predict():
    data = {
        "monthly_savings_consistency": [float(request.args["monthly_savings_consistency"])],
        "rent_and_bill_payment_history": [float(request.args["rent_and_bill_payment_history"])],
        "spending_to_income_ratio": [float(request.args["spending_to_income_ratio"])]
    }

    X = pd.DataFrame(data)
    prob = model.predict_proba(X)[:, 1]

    return f"Probability of loan repayment success: {prob[0]:.2f}"


