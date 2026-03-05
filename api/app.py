from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

MODEL_PATH = "models/churn_pipeline.joblib"

app = FastAPI(title="Customer Churn Prediction API", version="1.0")
pipe = joblib.load(MODEL_PATH)


class CustomerInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: CustomerInput):
    df = pd.DataFrame([payload.model_dump()])
    prob = float(pipe.predict_proba(df)[:, 1][0])
    pred = int(prob >= 0.5)

    return {"churn_probability": prob, "churn_prediction": pred}