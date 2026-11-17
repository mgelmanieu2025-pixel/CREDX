# CredX – Alternative Credit Score for Young People (MVP)

**Youth Fintech project · Assigned risk: Credit Risk**

Young people (students, first-job professionals, freelancers) often have **no traditional credit history**, so banks can't assess them. They also tend to fall into risky habits such as BNPL stacking and overdrafts, which lead to default. CredX scores credit readiness from **alternative data** and gives personalised advice to reduce default risk.

## Main user journey
1. Pick a profile: **Student**, **Young professional** or **Freelancer / gig worker**. You can also load a sample persona.
2. Answer 3 short steps (about 2 min):
   - Housing & bills: rent on time, rent history, bills on time
   - Savings & spending: income, savings rate, months saved in the last year, emergency fund, non-essential spending
   - Income & debt: income history, income volatility, gig rating (freelancers only), debt-to-income, BNPL plans, overdrafts
3. Get your results:
   - **Live loan repayment probability** from our Flask ML API on PythonAnywhere, with a decision
   - **CredX Readiness Score (300–850)** and a risk band
   - **Predicted 12-month default probability**, repayment reliability, loan readiness and a suggested safe starter-credit amount
   - **Default-prevention alerts**, for example BNPL stacking, overdrafts, high debt-to-income, or no emergency buffer
   - **Factors helping and hurting** your score, explained by the model's contributions
   - **Action plan** ranked by the number of score points the model predicts each step would add
   - **What-if simulator**: move sliders and watch the score update live
4. Your checks are saved on the device (*My checks*) so you can track progress. You can print or save a report as a PDF.

## Machine-learning models
### 1. Loan repayment API (Python / Flask on PythonAnywhere)
- Endpoint: `https://mirandagelman.pythonanywhere.com/predict?monthly_savings_consistency=0.80&rent_and_bill_payment_history=0.90&spending_to_income_ratio=0.30`
- Returns the **probability of loan repayment success** (e.g. `0.89`).
- The web app calls it through a Vercel serverless function, `api/predict.js`. The function validates the inputs, calls PythonAnywhere server-side (which avoids CORS) and returns JSON.
- How the user's answers become the 3 model inputs (all on a 0–1 scale):
  | API feature | Calculated from |
  |---|---|
  | monthly_savings_consistency | months with savings in the last 12 ÷ 12 |
  | rent_and_bill_payment_history | average of rent-on-time % and bills-on-time % ÷ 100 |
  | spending_to_income_ratio | 1 − savings rate |
- Decision bands: ≥ 75% likely to repay · 50–75% borderline · < 50% high risk.

### 2. In-app alternative credit score
- `model/train_model.py` trains a **logistic regression** (scikit-learn) on 20,000 synthetic young-borrower records, using 12 alternative-data features.
- Test performance: **AUC ≈ 0.81**, accuracy ≈ 0.83 (25% hold-out).
- The trained weights are exported to `model.json` and the scoring runs **in the browser**. No user data leaves the device.
- Probability of default is converted to points with standard scorecard scaling: score = 640 − 45·ln(odds of default), clamped to 300–850.

Retrain: `pip install scikit-learn numpy` then `python model/train_model.py`.

## Tech
Static HTML/CSS/JavaScript front end, plus one Vercel serverless function (`api/predict.js`) that connects to the Python ML API. There's no build step. On Vercel, set Framework preset to **Other**.

## Run locally
```
python -m http.server 8000
# open http://localhost:8000
```

> Educational MVP. This is not a lending decision or financial advice. The training data is synthetic.
