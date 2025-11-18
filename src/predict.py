import pickle 
from typing import Dict, Literal

import pandas as pd
import uvicorn
from fastapi import FastAPI

from pydantic import BaseModel, Field

app = FastAPI(title="Heart-disease-prediction")

# Request model
class Patient(BaseModel):
    ST_Slope: Literal["Up", "Flat", "Down"]
    ChestPainType: Literal["TA", "ATA", "NAP", "ASY"]
    ExerciseAngina: Literal["Y", "N"]
    Oldpeak: float = Field(..., ge=-2.6, le=6.2)
    MaxHR: int = Field(..., ge=60, le=202)


# Response model
class PredictResponse(BaseModel):
    heart_disease_probability: float
    heart_disease: bool


with open('model/model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)
    

def predict_single(patient_data: Dict[str, any]) -> float:
    features = ['ST_Slope', 'ChestPainType', 'ExerciseAngina', 'Oldpeak', 'MaxHR']
    df_patient = pd.DataFrame([patient_data], columns=features)
    predict_proba = pipeline.predict_proba(df_patient)[0, 1]
    return predict_proba

@app.post("/predict")
def predict(patient: Patient) -> PredictResponse:
    prob = predict_single(patient.model_dump())
    return PredictResponse(
        heart_disease_probability = prob,
        heart_disease =  prob >= 0.5
        
    )


if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    