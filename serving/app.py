from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/fraud_model.pkl")

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Real ML Fraud Detection System",
    version="2.0"
)

# Input Schema
class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    Amount: float

@app.get("/")
def home():
    return {
        "message": "Fraud Detection API Running Successfully"
    }

@app.post("/predict")
def predict(transaction: Transaction):

    # Convert input to DataFrame
    data = pd.DataFrame([transaction.dict()])

    # Add missing columns with default value
    required_columns = [
        'Time','V1','V2','V3','V4','V5',
        'V6','V7','V8','V9','V10',
        'V11','V12','V13','V14','V15',
        'V16','V17','V18','V19','V20',
        'V21','V22','V23','V24','V25',
        'V26','V27','V28','Amount'
    ]

    for col in required_columns:
        if col not in data.columns:
            data[col] = 0

    data = data[required_columns]

    # Prediction
    prediction = model.predict(data)[0]

    if prediction == 1:
        result = "⚠️ Fraudulent Transaction"
    else:
        result = "✅ Normal Transaction"

    return {
        "prediction": result
    }