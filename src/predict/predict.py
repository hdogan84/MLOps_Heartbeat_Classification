from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
from typing import List

app = FastAPI()
router = APIRouter()

class EKGSignal(BaseModel):
    signal: List[float]

@router.get("/status")
async def get_status():
    return {"status": "Prediction API is up"}


#this functions must be copied from the basic scripts.
@router.post("/predict")
async def make_prediction(ekg_signal: EKGSignal, model_name: str = "RFC_Mitbih"):
    # Lade das Modell aus MLFlow und mache eine Vorhersage
    return {"prediction": "dummy_result"}