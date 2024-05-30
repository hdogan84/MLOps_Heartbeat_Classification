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


async def send_prediction_request(model_name: str, dataset: str):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "http://predict-api:8003/predict",  # Using service name
                json={"model_name": model_name, "dataset": dataset}
            )
            response.raise_for_status()
            logging.info(f"Prediction request successful for model: {model_name}")
        except httpx.HTTPStatusError as exc:
            logging.error(f"HTTP error occurred: {exc.response.status_code} - {exc.response.text}")
        except httpx.RequestError as exc:
            logging.error(f"Request error occurred: {exc}")
        except Exception as exc:
            logging.error(f"Unexpected error occurred: {exc}")

@app.post("/predict_realtime", dependencies=[Depends(RateLimiter(times=180, seconds=60))]) #according to a max of 180 Heartbeats per second.
async def call_prediction_api(background_tasks: BackgroundTasks, model_name: str = "RFC", dataset: str = "Mitbih"):
    logging.info(f"Received prediction_realtime request with model: {model_name} and Dataset {dataset}")
    background_tasks.add_task(send_prediction_request, model_name, dataset)
    return {"message": "Prediction request received, processing in the background."}

async def send_training_request(dataset: str, model_name: str): #model params could be an argument here...
    async with httpx.AsyncClient(timeout=360) as client: #setting the timeout to 360s to give the training endpoint enough time to finish and to avoid the "false" request error.
        try:
            response = await client.post(
                "http://train-api:8001/train",  # Using service name
                json={"dataset": dataset, "model_name": model_name, "model_params": {}}
            )
            response.raise_for_status()
            logging.info(f"Training request successful for model: {model_name} with dataset: {dataset}")
        except httpx.HTTPStatusError as exc:
            logging.error(f"HTTP error occurred: {exc.response.status_code} - {exc.response.text}")
        except httpx.RequestError as exc:
            logging.error(f"Request error occurred: {exc}")
        except Exception as exc:
            logging.error(f"Unexpected error occurred: {exc}")

@app.post("/train", dependencies=[Depends(RateLimiter(times=5, seconds=60))])
async def call_training_api(background_tasks: BackgroundTasks, dataset: str = "Ptbdb", model_name: str = "RFC"):
    background_tasks.add_task(send_training_request, dataset, model_name)
    return {"message": "Training request received, processing in the background."}

async def send_update_request(model_name: str, metric_name: str):
    async with httpx.AsyncClient(timeout=180) as client: #setting the timeout to 180s to give the training endpoint enough time to finish and to avoid the "false" request error.
        try:
            response = await client.post(
                "http://update-api:8002/update",
                json={"model_name": model_name, "metric_name": metric_name}
            )
            response.raise_for_status()
            logging.info(f"Successfully updated model {model_name} with metric {metric_name}")
        except httpx.HTTPStatusError as exc:
            logging.error(f"HTTP error occurred: {exc.response.status_code} - {exc.response.text}")
        except httpx.RequestError as exc:
            logging.error(f"Request error occurred: {exc}")
        except Exception as exc:
            logging.error(f"Unexpected error occurred: {exc}")

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
