from fastapi import FastAPI

app = FastAPI()

#copy most of the code from previous functions from master branch.
@app.post("/train")
def make_prediction(data: dict):
    # train a model with the selected parameters and training dataset
    return {"train": "dummy_model"}