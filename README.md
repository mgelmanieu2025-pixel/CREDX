# CredX

CredX is a fintech app that helps young adults estimate whether they are financially ready for a loan using alternative financial behavior data instead of traditional credit history.

## Problem
Young adults often do not have enough credit history, which makes it difficult for banks to judge whether they are good borrowers.

## Machine Learning Task
Supervised classification using logistic regression.

## Features
- monthly_savings_consistency
- rent_and_bill_payment_history
- spending_to_income_ratio

## Target Variable
- loan_repayment_success
  - 1 = good borrower / repaid the loan on time
  - 0 = risky borrower / missed payments or defaulted

## Files
- `fintech.py` - code that generates synthetic data, trains the model, evaluates it, and saves the model
- `fintech.xlsx` - synthetic dataset
- `fintech.joblib` - trained model
- `flask_app.py` - web application and API
- `templates/index.html` - HTML front end
- `requirements.txt` - Python dependencies

## How to Run
1. Install dependencies:
   pip install -r requirements.txt

2. Run the Flask app:
   python flask_app.py

## PythonAnywhere Deployment
Public URL:
https://mirandagelman.pythonanywhere.com

## HTML Front End
The web page allows users to enter:
- monthly_savings_consistency
- rent_and_bill_payment_history
- spending_to_income_ratio

It returns a live prediction and prediction probability.

## API Endpoint
POST /api/predict

Full API URL:
https://mirandagelman.pythonanywhere.com/api/predict

### Example request
curl -X POST https://mirandagelman.pythonanywhere.com/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "monthly_savings_consistency": 0.8,
    "rent_and_bill_payment_history": 0.9,
    "spending_to_income_ratio": 0.3
  }'

### Example response
{
  "prediction": 1,
  "label": "good borrower / loan repayment success",
  "probability": 0.89
}

## Notes
- The HTML form and the API endpoint both use the trained `fintech.joblib` model.
- This project demonstrates how alternative financial behavior data can help estimate loan readiness for young adults.
