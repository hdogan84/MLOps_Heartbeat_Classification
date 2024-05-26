from fastapi import FastAPI, APIRouter

app = FastAPI()
router = APIRouter() #if the router.get doesnt work, this is obsolete?


#@router.get("/status") #this does not work?
@app.get("/status")
async def get_status():
    return {"status": "Update API is up"}

#copy the code from the master branch, is essentially the set_deployment_alias function and nothing more.
#@router.post("/update") #this doesnt work
@app.post("/update")
async def update_deployment_model(model_name: str = "RFC_Mitbih", metric_name: str = "accuracy"):
    """
    The actual updating code from backup_codes. Just a dummy for now.
    """
    # execute the set_deployment_alias() function
    return {"deployment_model_set": "True + Version"}