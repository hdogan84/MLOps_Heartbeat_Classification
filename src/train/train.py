from fastapi import FastAPI, APIRouter

app = FastAPI()
router = APIRouter()


@router.get("/status")
async def get_status():
    return {"status": "Training API is up"}
#copy most of the code from previous functions from master branch.

@router.post("/train")
async def train_model(dataset: str, model_name: str):
    """
    The actual training code from backup_codes. Just a dummy for now.
    """
    return {"trained": True, "model_name": model_name}