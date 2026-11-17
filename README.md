# CredX – Alternative Credit Score for Young People (MVP)

**Youth Fintech project · Assigned risk: Credit Risk**

Young people (students, first-job professionals, freelancers) often have **no traditional credit history**, so banks can't assess them. They also tend to fall into risky habits such as BNPL stacking and overdrafts, which lead to default. CredX scores credit readiness from **alternative data** and gives personalised advice to reduce default risk.

## Main user journey
1. Pick a profile: **Student**, **Young professional** or **Freelancer / gig worker**. You can also load a sample persona.
2. Answer 3 short steps (about 2 min):
   - Housing & bills: rent on time, rent history, bills on time
   - Savings & spending: income, savings rate, emergency fund, non-essential spending
   - Income & debt: income history, income volatility, gig rating (freelancers only), debt-to-income, BNPL plans, overdrafts
3. Get your results:
   - **CredX Readiness Score (300–850)** and a risk band
   - **Predicted 12-month default probability**, repayment reliability, loan readiness and a suggested safe starter-credit amount
   - **Default-prevention alerts**, for example BNPL stacking, overdrafts, high debt-to-income, or no emergency buffer
   - **Factors helping and hurting** your score, explained by the model's contributions
   - **Action plan** ranked by the number of score points the model predicts each step would add
   - **What-if simulator**: move sliders and watch the score update live
4. Your checks are saved on the device (*My checks*) so you can track progress. You can print or save a report as a PDF.

## Machine-learning model
- `model/train_model.py` trains a **logistic regression** (scikit-learn) on 20,000 synthetic young-borrower records, using 12 alternative-data features.
- Test performance: **AUC ≈ 0.81**, accuracy ≈ 0.83 (25% hold-out).
- The trained weights are exported to `model.json` and the scoring runs **in the browser**. No user data leaves the device.
- Probability of default is converted to points with standard scorecard scaling: score = 640 − 45·ln(odds of default), clamped to 300–850.

Retrain: `pip install scikit-learn numpy` then `python model/train_model.py`.

## Tech
Static HTML/CSS/JavaScript with no build step. It deploys to Vercel as-is (Framework preset: **Other**, no build command).

## Run locally
```
python -m http.server 8000
# open http://localhost:8000
```

> Educational MVP. This is not a lending decision or financial advice. The training data is synthetic.
