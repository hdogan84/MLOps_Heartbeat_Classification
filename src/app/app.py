##### IMPORTANT: THis version of the app.py script is written for v1: Only one gateway api, therefore all scripts are included in the /app folder (Duplicates in the other folders))


import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent)) #this might be unnecessary for V1, try to leave it out.
print(sys.path)
import os

from fastapi import FastAPI, Query, Depends, HTTPException
from contextlib import asynccontextmanager
from user_db import User, create_db_and_tables
from user_schemas import UserCreate, UserRead, UserUpdate
from users import auth_backend, current_active_user, fastapi_users
from make_dataset_V1 import download_datasets, prepare_datasets #V1!!!
from model_functions_V1 import load_ml_model, predict_with_ml_model #V1!!!
from app_functions import select_random_row, register_model, set_deployment_alias, load_deployment_model
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RFC, BaggingClassifier as BG
from sklearn.svm import SVC as SVC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.metrics import accuracy_score
#import xgboost as XGB
from tqdm import tqdm 

import mlflow
import mlflow.sklearn
#V1: Tracking Uri is set as environment variable in docker-compose.yaml
#mlflow.set_tracking_uri("file:///app/mlruns")#V1 for Docker--> This must be path inside the container!
#mlflow.set_tracking_uri("./mlruns") #old code
experiment_name = "debugging_experiment"
mlflow.set_experiment(experiment_name)
from mlflow.tracking import MlflowClient
client = MlflowClient() #define the client after setting the tracking uri, otherwise a not used mlruns directory will be created in the app folder (undesirable)

#as bash (for docker-compose.yml later?)
#export MLFLOW_TRACKING_URI="/home/simon/May24_MLOps_Heartbeat_Classification/mlruns"

## Side note: If you cannot import the selfwritten modules, this might help, especially when working with venv: https://stackoverflow.com/questions/71754064/vs-code-pylance-problem-with-module-imports


from typing import List, Dict
import pandas as pd
import json
from pydantic import BaseModel
import uvicorn
import logging
from pathlib import Path

# Define the path for the log file
log_file_path = Path("reports/logs/app.log") #V1: We put the logs directly in some folder placed in /app, hopefully it will be created inside the docker-container.
log_file_path.parent.mkdir(parents=True, exist_ok=True) # Ensure the directory exists
if not log_file_path.exists(): #Ensure the log file exists
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


global dataset_cache #global variable for cached test and train datasets (must not be created each time a function is called)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Not needed if you setup a migration system like Alembic
    await create_db_and_tables()
    yield


app = FastAPI(lifespan=lifespan)
#app = FastAPI() #old code without authentifciation


###### HERE SOME AUTHENTIFICATION ROUTES ARE IMPLEMENTED #########
# CODE COPIED FROM https://fastapi-users.github.io/fastapi-users/10.1/configuration/full-example/

app.include_router(
    fastapi_users.get_auth_router(auth_backend), prefix="/auth/jwt", tags=["auth"]
)
app.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),
    prefix="/auth",
    tags=["auth"],
)
app.include_router(
    fastapi_users.get_reset_password_router(),
    prefix="/auth",
    tags=["auth"],
)
app.include_router(
    fastapi_users.get_verify_router(UserRead),
    prefix="/auth",
    tags=["auth"],
)
app.include_router(
    fastapi_users.get_users_router(UserRead, UserUpdate),
    prefix="/users",
    tags=["users"],
)

######### THIS IS THE EXAMPLE FOR TESTING THE USER ROLES (Only superuser == admin and normal user available)
@app.get("/authenticated-route", tags=["auth"])
async def authenticated_route(user: User = Depends(current_active_user)):
    if user.is_superuser:
        return {"message": f"Hello {user.email}, you are a superuser"}
    else:
        return {"message": f"Hello {user.email}, you are not a superuser"}

# Placeholder for dataset names (and links)  --> Should be a file that is growable, for now only hardcoding
datasets = {
    "Mitbih_test": "../data/heartbeat/mitbih_test.csv",
    "Mitbih_train": "../data/heartbeat/mitbih_train.csv",

    "Ptbdb_test": "../data/heartbeat/ptbdb_test.csv",
    "Ptbdb_train": "../data/heartbeat/ptbdb_train.csv"
}

# Placeholder for notifications  --> Should be a file that is growable, for now only hardcoding
notifications = []

# Endpoint to check API status
@app.get("/status")
async def get_status():
    return {"status": 1}

# Endpoint to predict an EKG signal in real-time. For now manually triggered and fed with random row for simulation purposes.
class EKGSignal(BaseModel):
    signal: List[float]

@app.post("/predict_realtime")
async def predict_realtime(ekg_signal: EKGSignal, model_name: str = "RFC_Mitbih"):
    """
    Endpoint to predict a random row which simulates the real world application
    ekg_signal: The output format of the response body
    model_name: the desired model to use (deployment model will be searched on mlflow webserver)

    Functionality:
    (1): load the deployment model according to model name frin mlflow webserver
    (2): predict with the deployment model
    """
    logging.info(f"Received prediction_realtime request with model: {model_name}")

    try:
        # Load model from MLflow registry
        ml_model_deployed = load_deployment_model(model_name=model_name)
        logging.info("ml_Model loaded successfully with our load_deployment_model function")

        # Extract dataset name from model name
        dataset_name = model_name.split("_")[-1] + "_test" #works now, but source for errors if naming convention changes.
        if dataset_name not in datasets:
            logging.error(f"Dataset {dataset_name} not found in datasets dictionary")
            return {"error": "Dataset not found"}

        data_path = "../data/"
        download_datasets(data_path)
        logging.info(f"Datasets downloaded to {data_path}")
      
        dataset_path = "../data/heartbeat/"
        cached_datasets = prepare_datasets(dataset_path)
        logging.info(f"Datasets prepared from {dataset_path}")

        X_test = cached_datasets[f"X_test_{model_name.split('_')[-1]}"]
        y_test = cached_datasets[f"y_test_{model_name.split('_')[-1]}"]

        rand_row, rand_target = select_random_row(X_test=X_test, y_test=y_test)
        logging.info("Random row selected from test data")

        with mlflow.start_run():
            logging.info("mlflow.start_run() entered")
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("dataset_name", dataset_name)

            if isinstance(rand_row, pd.Series):
                rand_row = rand_row.values.reshape(1, -1)
            elif isinstance(rand_row, np.ndarray):
                rand_row = rand_row.reshape(1, -1)
            logging.info("rand_row successfully prepared and beginning to predict. Rand row debug print:", rand_row)
            prediction = predict_with_ml_model(ml_model=ml_model_deployed, X=rand_row)
            logging.info("predictions with ML-Model from deployment made successfully")
            
            prediction_result = {
                "prediction": prediction.tolist() if isinstance(prediction, np.ndarray) else prediction
            }

            mlflow.log_param("input_data", rand_row.tolist() if isinstance(rand_row, np.ndarray) else rand_row.to_dict())
            mlflow.log_param("true_value", rand_target.tolist() if isinstance(rand_target, np.ndarray) else rand_target)

            if isinstance(prediction_result["prediction"], list):
                mlflow.log_param("predicted_value", prediction_result["prediction"][0])
            else:
                mlflow.log_param("predicted_value", prediction_result["prediction"])

        logging.info(f"Prediction successful: {prediction_result}")
        logging.info(f"True value: {rand_target}")
        logging.info("-------------------------------------------------------------------------------------------------------------")
        return {"prediction": prediction_result, "true_value": int(rand_target)}

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return {"error": str(e)}
    

# Endpoint to retrain a model on a new dataset
#RETRAIN AND UPDATE CAN BE ONE ENDPOINT IF MLFLOW IS USED! Or just remove retrrain for the ML Models and use it only for DL models if at all.
@app.post("/retrain")
async def retrain_model(dataset: str, model_name: str):
    if model_name not in models["Classifiers"]:
        return {"error": "Model option not available"}
    else: 
        logging.info(f"Received train request with model: {model_name}")
        # Add some extra stuff here such as "existing model might be overwriten! "
@app.post("/train")
async def train_model_mlflow(dataset: str = "Ptbdb", model_name: str = "RFC"):
    """
    Endpoint to train and register new models or new versions of existing models.
    Creates a new instance of the specified model and trains it on the selected dataset.
    Then registration (logging) on the MLFlow Webserver is done, without automatic setting of the best version as deployment model for this model_name.

    dataset: the dataset_name as string
    model_name: the model_name as string

    Future work:
    - use params_dict as argument to refine the model_training (same model_name)
    """
    
    if "RFC" in model_name: #Simon: This is more inclusive
        model = RFC() #simon: Here some params in form of a dict could be passed.
    logging.info(f"Initiated {model_name} trainer") #simon: and the params could be logged.

    dataset_name = f"{dataset}_train"

    if dataset_name not in datasets:
        logging.error(f"Dataset {dataset_name} not found")
        return {"error": "Dataset not found"}

    try:
        data_path = "../data/"
        download_datasets(data_path)
        logging.info(f"Datasets downloaded to {data_path}")
      
        dataset_path = "../data/heartbeat/"
        cached_datasets = prepare_datasets(dataset_path)
        logging.info(f"Datasets prepared from {dataset_path}")

        X_train = cached_datasets[f"X_train_{dataset}"]
        y_train = cached_datasets[f"y_train_{dataset}"]

        X_test = cached_datasets[f"X_test_{dataset}"]
        y_test = cached_datasets[f"y_test_{dataset}"]
    
        logging.info("Data load successful")
        logging.info("----------------------------------------------------------")

    except Exception as e:
        logging.error(f"Error during data loading: {e}")
        return {"error": "Dataload failed"}
    
    
    model.fit(X_train, y_train)

    logging.info(f"{model_name} model train successful")

    # Evaluate model
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    metrics = {"accuracy": acc} #more metrics could be appended, like confusion matrix and so on.

    # Save the new model and log metrics (dummy response here)
    new_model_name = model_name + "_" + dataset #simon: this follows the naming convention used in project 1
    
    # Store information in tracking server
    with mlflow.start_run() as run: #simon: removed (run_name=run_name) as argument to stay in one structure for debugging.
        mlflow.log_params({"dataset": dataset, "model_name": new_model_name})
        mlflow.log_metrics(metrics)
        #simon: copied the model-registration code from function register_model()
        relative_model_path = os.path.relpath(new_model_name, start=os.getcwd()) #fetching the relative path for the model_name
        logging.info(f"relative_model_path from register_model(): {relative_model_path}")
        mlflow.sklearn.log_model(model, artifact_path=relative_model_path, registered_model_name=new_model_name)
        
        #mlflow.sklearn.log_model(sk_model=model, input_example=X_test, artifact_path=artifact_path) #simon: Deactivated for debugging.
    logging.info("------------------------------Model training successfull---------------------------------")
    return {"status": "trained", "model_name": new_model_name, "metrics": metrics} #simon: ommited the models dict for training: model_metrics[new_model_name]

# Endpoint to update the production model
@app.post("/update_model")
async def update_model(model_name: str = "RFC_Mitbih", metric_name: str = "accuracy"):
    """
    Endpoint to just execute the set_deployment_alias function.
    This function just searches in the mlflow model_registry for all models with the model_name and sets the one 
    with the best accuracy score to the new deployment model. 

    model_name: the name of the model, default / debugging value is 'RFC_Mitbih'.
    metric_name: the metric on which the models with the same model name shall be compared, default value is accuracy.
    """
    set_deployment_alias(model_name=model_name, metric_name=metric_name)
    logging.info(f"Model alias set to deployment based on the best accuracy")
    #if model_name not in models: #simon: This is not necessary if we use the mlflow model_registry
    #    return {"error": "Model not found"}
    
    return {"status": "updated", "model_name": model_name}

# Endpoint to monitor current production model
@app.get("/monitor")
async def monitor(classifier: str, dataset: str):
    # We could skip the resampling option in the API, but return the better performing model
    # For example, return RF_Oversampled_PTB metric, with an additional key: result[dataset]: PTB_SMOTE_B etc.

    """
    Select Machine Learning Models from (in string format):
    SVM, KNN, XGB, DTC, or RFC

    Select Dataset from the options:
    MITBIH or PTBDB

    """

    # Retrieve metrics of the current production model
    # If there is no metrics available, the model must be used on the selected dataset and the report must be created (this essentially makes the predict_batch function useless!)
    reports_folder = "../../models/ML_Models/classification_reports/"
    clf_report = classifier + "_Basemodel_no_gridsearch_"+ str(dataset) +"_A_Original_classification_report.txt"
    df = pd.read_csv(reports_folder+clf_report, sep="\t")

    #metrics = model_metrics[production_model_name]
    # need to rearrange the DF for better visualisation in the response body

    # Log metrics with MLflow ---> THIS IS CODE TO BE COMPLETED, NOT WORKING!
    with mlflow.start_run():
        mlflow.log_param("model_name", new_model_name)
        mlflow.log_metrics(model_metrics[new_model_name])
        # Dummy code for logging model. Replace with actual model object.
        mlflow.sklearn.log_model(None, new_model_name)  # Replace None with actual model
    
    return df

# Endpoint to monitor all models
@app.get("/monitor_all")
async def monitor_all():
    return model_metrics

# Endpoint to send notifications to medical staff
class Notification(BaseModel):
    email: str
    message: str

@app.post("/notify")
async def notify(notification: Notification):
    # Send notification (dummy response here)
    notifications.append(notification.dict())
    return {"status": "notification sent", "notification": notification.dict()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

