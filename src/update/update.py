from fastapi import FastAPI

app = FastAPI()

#copy the code from the master branch, is essentially the set_deployment_alias function and nothing more.
@app.post("/update")
def make_prediction(data: dict):
    # execute the set_deployment_alias() function
    return {"deployment_model_set": "True + Version"}