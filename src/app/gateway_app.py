import contextlib 
#import asyncio
#from uuid import uuid4
from pathlib import Path
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi_users.exceptions import UserAlreadyExists
from contextlib import asynccontextmanager
from user_db import User, create_db_and_tables, get_async_session, get_user_db
from user_schemas import UserCreate, UserRead, UserUpdate
from users import auth_backend, current_active_user, fastapi_users, get_user_manager
from pydantic import BaseModel
import uvicorn
import logging
import httpx
from typing import List
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from redis.asyncio import Redis

import json
import asyncio

##### TO-DOs: ######
# Necessary to add a pydantic for predict_sample function
class PredictModelRequest(BaseModel):
    model_name: str = "RFC"
    dataset: str = "Mitbih"
    x_sample: List = 187 * [0.1]

# Logging setup
log_file_path = Path("reports/logs/app.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Gateway")

### USEFUL LINKS ###
    # https://github.com/fastapi-users/fastapi-users/discussions/1361#discussioncomment-8661055 -_> Make nice gui and authentification process happen
    # https://fastapi-users.github.io/fastapi-users/10.3/configuration/authentication/transports/bearer/ --> Must define a bearer transport for the endpoints to work?


### Create context managers for creating a super user on startup ####
get_async_session_context = contextlib.asynccontextmanager(get_async_session)
get_user_db_context = contextlib.asynccontextmanager(get_user_db)
get_user_manager_context = contextlib.asynccontextmanager(get_user_manager)

### Function to create users per direct manipulation of the database ###
async def create_user(email: str, password: str, is_superuser: bool = False):
    try:
        async with get_async_session_context() as session:
            async with get_user_db_context(session) as user_db:
                async with get_user_manager_context(user_db) as user_manager:
                    user = await user_manager.create(
                        UserCreate(
                            email=email, password=password, is_superuser=is_superuser
                        )
                    )
                    print(f"User created {user}")
    except UserAlreadyExists:
        print(f"User {email} already exists")

### Function to check if the current user is a superuser (for protection).
def current_active_superuser(user: User = Depends(current_active_user)):
    if not user.is_superuser:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    return user

##### TO-DOs: ######

#this solution works and doesnt get a warning for depreceated use of @app.on_event("startup"), but is not written as example in the official redis page!
@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_db_and_tables()
    await create_user("admin@example.com", "admin", is_superuser=True) #creation of the superuser on startup
    redis = Redis(host='redis', port=6379, decode_responses=True) #redis might lead to bugging and not closing the containers correctly, therefore a try structure is established.
    await FastAPILimiter.init(redis)
    logger.info("Created all initiations for the lifespan in async def lifespan")
    try:
        yield
    finally:
        await redis.close()
        logger.info("Redis connection closed")

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


async def send_data_simulation_request(model_name: str, dataset: str):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "http://data-simulation-api:8004/data_simulation",  # Using service name
                json={"model_name": model_name, "dataset": dataset}
            )
            response.raise_for_status()
            logger.info(f"Data simulation request successful for model: {model_name}")
        except httpx.HTTPStatusError as exc:
            logger.error(f"HTTP error occurred: {exc.response.status_code} - {exc.response.text}")
        except httpx.RequestError as exc:
            logger.error(f"Request error occurred: {exc}")
        except Exception as exc:
            logger.error(f"Unexpected error occurred: {exc}")

    logger.info(f"Data API Response is: {response.json()}")

    data_response = json.loads(response.json()) # json load is a needed to transform the data back to dict 
    x_sample = data_response["x_sample"]

    # call the async def function that is already implemented
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "http://predict-sample-api:8005/predict_sample",  # Using service name
                json={"model_name": model_name, "dataset": dataset, "x_sample": x_sample}
            )
            response.raise_for_status()
            logger.info(f"Prediction sample request successful for model: {model_name}")
        
        except httpx.HTTPStatusError as exc:
            logger.error(f"HTTP error occurred: {exc.response.status_code} - {exc.response.text}")
        except httpx.RequestError as exc:
            logger.error(f"Request error occurred: {exc}")
        except Exception as exc:
            logger.error(f"Unexpected error occurred: {exc}")


@app.post("/data_simulation", dependencies=[Depends(RateLimiter(times=180, seconds=60))])
async def call_data_simulation_api(background_tasks: BackgroundTasks, model_name: str = "RFC", dataset: str = "Mitbih"):
    logger.info(f"Received data simulation & prediction request with model: {model_name} and Dataset {dataset}")

    for _ in range(5):  # 60 tasks
        background_tasks.add_task(send_data_simulation_request, model_name, dataset)
        # import asyncio and put it in requirements.txt or just use time.sleep(1) instead.
        #await asyncio.sleep(1)  # wait 1 second between each task

    #background_tasks.add_task(schedule_tasks)
    return {"message": "Data request received, processing in the background."}


async def send_predict_sample_request(model_name: str, dataset: str, x_sample):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "http://predict-sample-api:8005/predict_sample",  # Using service name
                json={"model_name": model_name, "dataset": dataset, "x_sample": x_sample}
            )
            response.raise_for_status()
            logger.info(f"Prediction sample request successful for model: {model_name}")
        
        except httpx.HTTPStatusError as exc:
            logger.error(f"HTTP error occurred: {exc.response.status_code} - {exc.response.text}")
        except httpx.RequestError as exc:
            logger.error(f"Request error occurred: {exc}")
        except Exception as exc:
            logger.error(f"Unexpected error occurred: {exc}")

@app.post("/predict_sample", dependencies=[Depends(RateLimiter(times=180, seconds=60))]) #according to a max of 180 Heartbeats per second.
async def call_predict_sample_api(background_tasks: BackgroundTasks, item: PredictModelRequest):
    model_name = item.model_name
    dataset = item.dataset
    x_sample = item.x_sample
    logger.info(f"Received predict_sample request with model: {model_name} and Dataset {dataset}")
    background_tasks.add_task(send_predict_sample_request, model_name, dataset, x_sample)
    return {"message": "Prediction request received, processing in the background."}

async def send_prediction_request(model_name: str, dataset: str):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "http://predict-api:8003/predict",  # Using service name
                json={"model_name": model_name, "dataset": dataset}
            )
            response.raise_for_status()
            logger.info(f"Prediction request successful for model: {model_name}")
        except httpx.HTTPStatusError as exc:
            logger.error(f"HTTP error occurred: {exc.response.status_code} - {exc.response.text}")
        except httpx.RequestError as exc:
            logger.error(f"Request error occurred: {exc}")
        except Exception as exc:
            logger.error(f"Unexpected error occurred: {exc}")

@app.post("/predict_realtime", dependencies=[Depends(RateLimiter(times=180, seconds=60))]) #according to a max of 180 Heartbeats per second.
async def call_prediction_api(background_tasks: BackgroundTasks, model_name: str = "RFC", dataset: str = "Mitbih"):
    logger.info(f"Received prediction_realtime request with model: {model_name} and Dataset {dataset}")
    background_tasks.add_task(send_prediction_request, model_name, dataset)
    return {"message": "Prediction request received, processing in the background."}

async def send_training_request(dataset: str, model_name: str, model_params: dict =  {"n_jobs": -1}): #model params could be an argument here...
    async with httpx.AsyncClient(timeout=360) as client: #setting the timeout to 360s to give the training endpoint enough time to finish and to avoid the "false" request error.
        try:
            response = await client.post(
                "http://train-api:8001/train",  # Using service name
                json={"dataset": dataset, "model_name": model_name, "model_params": model_params}
            )
            response.raise_for_status()
            logger.info(f"Training request successful for model: {model_name} with dataset: {dataset}")
        except httpx.HTTPStatusError as exc:
            logger.error(f"HTTP error occurred: {exc.response.status_code} - {exc.response.text}")
        except httpx.RequestError as exc:
            logger.error(f"Request error occurred: {exc}")
        except Exception as exc:
            logger.error(f"Unexpected error occurred: {exc}")

@app.post("/train", dependencies=[Depends(RateLimiter(times=5, seconds=60)), Depends(current_active_superuser)])
async def call_training_api(background_tasks: BackgroundTasks, dataset: str = "Ptbdb", model_name: str = "RFC", model_params: dict = {"n_jobs": -1}):
    background_tasks.add_task(send_training_request, dataset, model_name, model_params)
    return {"message": "Training request received, processing in the background."}

async def send_update_request(model_name: str, dataset: str, metric_name: str):
    async with httpx.AsyncClient(timeout=180) as client: #setting the timeout to 180s to give the training endpoint enough time to finish and to avoid the "false" request error.
        try:
            response = await client.post(
                "http://update-api:8002/update",
                json={"model_name": model_name, "dataset": dataset, "metric_name": metric_name}
            )
            response.raise_for_status()
            logger.info(f"Successfully updated model {model_name} on dataset {dataset} with metric {metric_name}")
        except httpx.HTTPStatusError as exc:
            logger.error(f"HTTP error occurred: {exc.response.status_code} - {exc.response.text}")
        except httpx.RequestError as exc:
            logger.error(f"Request error occurred: {exc}")
        except Exception as exc:
            logger.error(f"Unexpected error occurred: {exc}")

@app.post("/update_model", dependencies=[Depends(RateLimiter(times=5, seconds=60)), Depends(current_active_superuser)])
async def call_update_api(background_tasks: BackgroundTasks, model_name: str = "RFC", dataset: str = "Ptbdb", metric_name: str = "accuracy"):
    background_tasks.add_task(send_update_request, model_name, dataset, metric_name)
    return {"message": "Update request received, processing in the background."}


class Notification(BaseModel):
    email: str
    message: str

@app.post("/notify")
async def call_notification_api(notification: Notification):
    notifications.append(notification.dict())
    return {"status": "notification sent", "notification": notification.dict()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
