from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

class EKGSignal(BaseModel):
    signal: List[float]

@app.get("/status")
async def get_status():
    return {"status": "Prediction API is up"}


#this functions must be copied from the basic scripts.
@app.post("/predict")
async def make_prediction(ekg_signal: EKGSignal, model_name: str = "RFC_Mitbih"):
    # Lade das Modell aus MLFlow und mache eine Vorhersage
    return {"prediction": "dummy_result"}