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

from flask import jsonify

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)

    X = pd.DataFrame([[
        float(data["monthly_savings_consistency"]),
        float(data["rent_and_bill_payment_history"]),
        float(data["spending_to_income_ratio"])
    ]], columns=[
        "monthly_savings_consistency",
        "rent_and_bill_payment_history",
        "spending_to_income_ratio"
    ])

    prob = model.predict_proba(X)[0, 1]
    pred = int(model.predict(X)[0])

    return jsonify({
        "prediction": pred,
        "label": "good borrower / loan repayment success" if pred == 1 else "risky borrower / low repayment readiness",
        "probability": round(float(prob), 4)
    })
