from pathlib import Path
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from contextlib import asynccontextmanager
from user_db import User, create_db_and_tables
from user_schemas import UserCreate, UserRead, UserUpdate
from users import auth_backend, current_active_user, fastapi_users
from pydantic import BaseModel
import uvicorn
import logging
import httpx
from typing import List
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from redis.asyncio import Redis

##### TO-DOs: ######

# Define the path for the log file. This code is the same in each container, the log file is a bind mount defined in the docker-compose.yaml --> All containers write in this bind mount log file.
log_file_path = Path("reports/logs/app.log") # V1: We put the logs directly in some folder placed in /app, hopefully it will be created inside the docker-container.
log_file_path.parent.mkdir(parents=True, exist_ok=True) # Ensure the directory exists
if not log_file_path.exists(): # Ensure the log file exists
    log_file_path.touch()

# Configure the logging to write to the specified file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='a'),  # 'a' for append, 'w' for overwrite
        logging.StreamHandler()  # This will also print to console
    ]
)

#this solution works and doesnt get a warning for depreceated use of @app.on_event("startup"), but is not written as example in the official redis page!
@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_db_and_tables()
    redis = Redis(host='redis', port=6379, decode_responses=True) #redis might lead to bugging and not closing the containers correctly, therefore a try structure is established.
    await FastAPILimiter.init(redis)
    logging.info("Created all initiations for the lifespan in async def lifespan")
    try:
        yield
    finally:
        await redis.close()
        logging.info("Redis connection closed")

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

@app.post("/predict_realtime", dependencies=[Depends(RateLimiter(times=5, seconds=60))])
async def call_prediction_api(ekg_signal: EKGSignal, model_name: str = "RFC_Mitbih"):
    logging.info(f"Received prediction_realtime request with model: {model_name}")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "http://predict-api:8003/predict",  # Using service name
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

@app.post("/retrain", dependencies=[Depends(RateLimiter(times=5, seconds=60))])
async def call_retraining_api(dataset: str, model_name: str):
    return {"Retraining Success:": "True, Model XXX retrained."}


##### MAKE THE TRAIN ENDPOINT AND ALL OTHER RELEVANT ENDPOINTS WORK WITH BACKGROUND TASKS JUST LIKE THE UPDATE ENDPOINT!!!################ 
@app.post("/train", dependencies=[Depends(RateLimiter(times=5, seconds=60))])
async def call_training_api(dataset: str = "Ptbdb", model_name: str = "RFC"):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "http://train-api:8001/train",  # Using service name
                json={"dataset": dataset, "model_name": model_name, "model_params": {}}
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            logging.error(f"HTTP error occurred: {exc.response.status_code} - {exc.response.text}")
            raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
        except httpx.RequestError as exc:
            logging.error(f"Request error occurred: {exc}")
            raise HTTPException(status_code=500, detail="Internal Server Error")

async def send_update_request(model_name: str, metric_name: str):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "http://update-api:8002/update",
                json={"model_name": model_name, "metric_name": metric_name}
            )
            response.raise_for_status()
            logging.info(f"Successfully updated model {model_name} with metric {metric_name}")
        except httpx.HTTPStatusError as exc:
            logging.error(f"Request error occurred: {exc.response.status_code} - {exc.response.text}")
        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")

@app.post("/update_model", dependencies=[Depends(RateLimiter(times=5, seconds=60))])
async def call_update_api(background_tasks: BackgroundTasks, model_name: str = "RFC_Mitbih", metric_name: str = "accuracy"):
    background_tasks.add_task(send_update_request, model_name, metric_name)
    return {"message": "Update request received, processing in the background."}

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
