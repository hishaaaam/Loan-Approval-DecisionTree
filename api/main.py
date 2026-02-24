from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np

app = FastAPI(title="Loan Approval Decision Tree API")

model = joblib.load("model/model.pkl")


# -----------------------------
# Request schema (VERY IMPORTANT for viva)
# -----------------------------
class LoanRequest(BaseModel):
    income: float = Field(..., gt=0)
    credit_score: float = Field(..., ge=300, le=850)
    loan_amount: float = Field(..., gt=0)
    age: int = Field(..., ge=18, le=100)
    employment_years: int = Field(..., ge=0)


@app.get("/")
def home():
    return {"message": "Loan Approval API is running"}


@app.post("/predict")
def predict(data: LoanRequest):
    try:
        features = np.array([[
            data.income,
            data.credit_score,
            data.loan_amount,
            data.age,
            data.employment_years
        ]])

        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0][pred]

        decision = "✅ Approved" if pred == 1 else "❌ Rejected"

        return {
            "prediction": int(pred),
            "decision": decision,
            "confidence": round(float(prob) * 100, 2)
        }

    except Exception as e:
        return {"error": str(e)}