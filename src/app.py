from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

# load model
model = joblib.load("models/heart_disease_model.pkl")
feature_names = joblib.load("models/feature_names.pkl")

class Patient(BaseModel):
    age: int
    trestbps: int
    chol: int
    thalch: int
    oldpeak: float 
    sex_Male: int 
    ca: int

@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API"}

@app.post("/predict")
def predict(patient: Patient):

    sample = pd.DataFrame(
        [[0.0] * len(feature_names)],
        columns=feature_names
    )

    data = patient.model_dump()

    for key, value in data.items():
        if key in sample.columns:
            sample.at[0, key] = float(value)

    prediction = model.predict(sample)

    return {
        "prediction": int(prediction[0]),
        "result": "Heart Disease" if prediction[0] == 1 else "No Heart Disease"
    }