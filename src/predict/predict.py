from fastapi import FastAPI

app = FastAPI()


#this functions must be copied from the basic scripts.
@app.post("/predict")
def make_prediction(data: dict):
    # Lade das Modell aus MLFlow und mache eine Vorhersage
    return {"prediction": "dummy_result"}