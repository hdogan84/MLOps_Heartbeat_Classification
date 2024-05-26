from pathlib import Path
from fastapi import FastAPI, Depends, HTTPException
from contextlib import asynccontextmanager
from user_db import User, create_db_and_tables
from user_schemas import UserCreate, UserRead, UserUpdate
from users import auth_backend, current_active_user, fastapi_users
from pydantic import BaseModel
import uvicorn
import logging
import httpx
from typing import List

# Define the path for the log file
log_file_path = Path("reports/logs/app.log")
log_file_path.parent.mkdir(parents=True, exist_ok=True)
if not log_file_path.exists():
    log_file_path.touch()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='a'),
        logging.StreamHandler()
    ]
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_db_and_tables()
    yield

app = FastAPI(lifespan=lifespan)

###### AUTHENTICATION ROUTES #########
app.include_router(
    fastapi_users.get_auth_router(auth_backend), prefix="/auth/jwt", tags=["auth"]
)
app.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate), prefix="/auth", tags=["auth"]
)
app.include_router(
    fastapi_users.get_reset_password_router(), prefix="/auth", tags=["auth"]
)
app.include_router(
    fastapi_users.get_verify_router(UserRead), prefix="/auth", tags=["auth"]
)
app.include_router(
    fastapi_users.get_users_router(UserRead, UserUpdate), prefix="/users", tags=["users"]
)

@app.get("/authenticated-route", tags=["auth"])
async def authenticated_route(user: User = Depends(current_active_user)):
    if user.is_superuser:
        return {"message": f"Hello {user.email}, you are a superuser"}
    else:
        return {"message": f"Hello {user.email}, you are not a superuser"}

# Placeholder for notifications
notifications = []

@app.get("/status")
async def get_status():
    return {"status": 1}

class EKGSignal(BaseModel):
    signal: List[float]

@app.post("/predict_realtime")
async def call_prediction_api(ekg_signal: EKGSignal, model_name: str = "RFC_Mitbih"):
    logging.info(f"Received prediction_realtime request with model: {model_name}")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "http://127.0.0.1:8003/predict",
                json={"signal": ekg_signal.signal, "model_name": model_name}
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            logging.error(f"HTTP error occurred: {exc.response.status_code} - {exc.response.text}")
            raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
        except httpx.RequestError as exc:
            logging.error(f"Request error occurred: {exc}")
            raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/retrain")
async def call_retraining_api(dataset: str, model_name: str):
    return {"Retraining Success:": "True, Model XXX retrained."}

@app.post("/train")
async def call_training_api(dataset: str = "Ptbdb", model_name: str = "RFC"):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "http://127.0.0.1:8001/train",
                json={"dataset": dataset, "model_name": model_name}
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            logging.error(f"HTTP error occurred: {exc.response.status_code} - {exc.response.text}")
            raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
        except httpx.RequestError as exc:
            logging.error(f"Request error occurred: {exc}")
            raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/update_model")
async def call_update_api(model_name: str = "RFC_Mitbih", metric_name: str = "accuracy"):
    async with httpx.AsyncClient() as client:
        try:
            #checking if the correct service name for the update api will fix the connection error.
            response = await client.post(
                "http://127.0.0.1:8002/update",
                json={"model_name": model_name, "metric_name": metric_name}
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            logging.error(f"HTTP error occurred: {exc.response.status_code} - {exc.response.text}")
            raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
        except httpx.RequestError as exc:
            logging.error(f"Request error occurred: {exc}")
            raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/monitor")
async def call_monitor_api(classifier: str, dataset: str):
    return {"DUMMY METRICS:", "DUMMY_VALUE"}

@app.get("/monitor_all")
async def call_monitor_all_api():
    return {"DUMMY METRICS FOR ALL MODELS:", "DUMMY_VALUES"}

class Notification(BaseModel):
    email: str
    message: str

@app.post("/notify")
async def call_notification_api(notification: Notification):
    notifications.append(notification.dict())
    return {"status": "notification sent", "notification": notification.dict()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
