import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
print(sys.path)


from fastapi import FastAPI, Query, Depends, HTTPException
from contextlib import asynccontextmanager
from user_db import User, create_db_and_tables
from user_schemas import UserCreate, UserRead, UserUpdate
from users import auth_backend, current_active_user, fastapi_users
from data.make_dataset import download_datasets, prepare_datasets
from models.model_functions import load_ml_model, predict_with_ml_model, load_advanced_cnn_model, predict_with_dl_model
from app_functions import select_random_row
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RFC, BaggingClassifier as BG
from sklearn.svm import SVC as SVC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier as DTC
#import xgboost as XGB

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
client = MlflowClient()
mlflow.set_tracking_uri("../../mlruns")
experiment_name = "debugging_experiment"
mlflow.set_experiment(experiment_name)
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
log_file_path = Path("../../reports/logs/app.log")
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

#@hakan, please see if you define the classifiers differently, in the model_registry only real models (.pkl / .h5) should be stored.
#"Classifiers": {"SVM": SVC, "RFC": RFC, "KNN": KNN, "DTC": DTC, "BG": BG}

# Placeholder model database --> Should be a file that is growable, for now only hardcoding
models = {
    "model_v1": {"path": "path/to/model_v1", "type": "ML", "num_classes": 2, "dataset": "Ptbdb"},
    "model_v2": {"path": "path/to/model_v2", "type": "ML", "num_classes": 2, "dataset": "Ptbdb"},
    "RFC_Mitbih_gridsearch": {"path": "../models/ML_Models/RFC_Optimized_Model_with_Gridsearch_MITBIH_A_Original.pkl", "type": "ML", "num_classes": 5, "dataset": "Mitbih"},
    "Best_DL_Model_Mitbih": {"path": "../models/DL_Models/Advanced_CNN/experiment_4_MITBIH_A_Original.weights.h5", "type": "DL_adv_cnn", "num_classes": 5, "dataset": "Mitbih"}
}

# Placeholder for metrics storage  --> Should be a file that is growable, for now only hardcoding
#Merge with models (registry)=???
#can also include classification report
model_metrics = {
    "model_v1": {"accuracy": 0.95, "confusion_matrix": [[50, 2], [1, 47]]},
    "model_v2": {"accuracy": 0.96, "confusion_matrix": [[51, 1], [2, 46]]}
}

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
async def predict_realtime(ekg_signal: EKGSignal, model_name: str = "Best_DL_Model_Mitbih"):
    #for docker containerization:
    # call the other predict_realtime api and pass the arguments
    # --> predict_realtime api is called with curl request or with requests (library)
    #model_name must be removed and used with tag "production" instead (if mlflow is used)
    
    logging.info(f"Received prediction_realtime request with model: {model_name}")

    if model_name not in models:
        logging.error(f"Model {model_name} not found")
        return {"error": "Model not found"}

    model_info = models[model_name]
    logging.info(f"Model info: {model_info}")
    model_path = model_info["path"]
    model_type = model_info["type"]
    num_classes = model_info["num_classes"]
    dataset_name = f"{model_info['dataset']}_test"

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

        X_test = cached_datasets[f"X_test_{model_info['dataset']}"]
        y_test = cached_datasets[f"y_test_{model_info['dataset']}"]

        rand_row, rand_target = select_random_row(X_test=X_test, y_test=y_test)
        logging.info(f"Random row selected from test data")

        with mlflow.start_run():
            logging.info("mlflow.start_run() entered") #rem,over later?
            mlflow.log_param("model_name", model_name)

            if model_type == "ML":
               
                #first load the model as .pkl file --> This is also in retraining / dumping in retraining necessary and here just a workaround.
                
                #the model loading from .pkl should not be done in production and in general not each time the endpoint is called.
                #ml_model = load_ml_model(model_path) #mlflow.sklearn not working correctly, but model does?
                #save the current loaded model in mlflow --> This is more for retraining / updating and here just a workaround
                
                #the log_model function creates a new version each time it is called, but only if the model is loaded from .pkl file.
                
                #just some basic mlflow logging for testing...
                
                #we have now stored some models on the webserver (this should be done with the updating / retraining endpoint --> They can be put together)
                #load the model --> The model_uri has to be known and is not existent yet.
                #filter_string = f"name='{model_name}'"
                #logging.info(f"filter_string: {filter_string}")
                #model_versions = client.search_model_versions(filter_string=filter_string)
                #model_versions = client.search_model_versions()
                #model_versions = client.get_model_version(name=model_name)
                #logging.info(f"model_versions for ML-Models in MLFlow: {model_versions}")
                #latest_version = max(model_versions, key=lambda mv: mv.version)
                #model_uri = f"../../mlruns//models:/{latest_version.name}/{latest_version.version}" #this is semi-optimal?
                model_uri = f"../../mlruns//models/{model_name}/version-3" #this is semi-optimal?
                ml_model = mlflow.sklearn.load_model(model_uri)

                #modelname is "RFC_Mitbih_gridsearch" in this debuggiong case, we should later use tags etc. with mlflow
                mlflow.sklearn.log_model(ml_model, artifact_path=model_name, registered_model_name=model_name) #This does save the models in the current directory although specified otherwise
                mlflow.log_param("model_path", model_path)

                
                
                logging.info("ml_Model loaded sucessfully with our mlflow.sklearn.load_model() function")
                if isinstance(rand_row, pd.Series):
                    rand_row = rand_row.values.reshape(1, -1)
                elif isinstance(rand_row, np.ndarray):
                    rand_row = rand_row.reshape(1, -1)
                logging.info("rand_row succesfully prepared and beginning to predict. Rand row debug print:", rand_row)
                prediction = predict_with_ml_model(ml_model=ml_model, X=rand_row)
                #ä##################################################################ATTENTION##########################
                #prediction = mflow.sklearn.predict() #if the mlflow.sklearn_model function works.
                logging.info("predictions with ML-Model made successfully")
            elif model_type == "DL_adv_cnn":
                dl_model = load_advanced_cnn_model(model_path=model_path, num_classes=num_classes)
                if isinstance(rand_row, pd.Series):
                    rand_row = rand_row.values.reshape(1, -1)
                elif isinstance(rand_row, np.ndarray):
                    rand_row = rand_row.reshape(1, -1)
                prediction = predict_with_dl_model(dl_model=dl_model, X=rand_row)
            else:
                logging.error(f"Unsupported model type: {model_info['type']}")
                return {"error": "Unsupported model type"}

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
        return {"prediction": prediction_result["prediction"]}
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return {"error": "Prediction failed"}


# Endpoint to retrain a model on a new dataset
#RETRAIN AND UPDATE CAN BE ONE ENDPOINT IF MLFLOW IS USED!
@app.post("/retrain")
async def retrain_model(dataset: str, model_name: str):
    if model_name not in models["Classifiers"]:
        return {"error": "Model option not available"}
    else: 
        logging.info(f"Received train request with model: {model_name}")
        # Add some extra stuff here such as "existing model might be overwriten! "

    model = models["Classifiers"][model_name]()
    logging.info(f"Initiated {model_name} trainer succesfully")
    
    # Load dataset, model, and perform retraining
    # data = pd.read_csv(dataset)
    # model = load_model(models[model_name])
    # new_model, new_metrics = retrain(model, data)
    
    # Save the new model and log metrics (dummy response here)
    new_model_name = model_name + "_retrained"
    models[new_model_name] = "path/to/new_model"
    model_metrics[new_model_name] = {"accuracy": 0.97, "confusion_matrix": [[52, 0], [1, 47]]}

    # Log metrics with MLflow ---> THIS IS CODE TO BE COMPLETED, NOT WORKING!
    with mlflow.start_run():
        mlflow.log_param("model_name", new_model_name)
        mlflow.log_metrics(model_metrics[new_model_name])
        # Dummy code for logging model. Replace with actual model object.
        mlflow.sklearn.log_model(None, new_model_name)  # Replace None with actual model
    
    return {"status": "retrained", "model_name": new_model_name, "metrics": model_metrics[new_model_name]}

# Endpoint to update the production model
@app.post("/update_model")
async def update_model(model_name: str):
    if model_name not in models:
        return {"error": "Model not found"}
    
    # Logic to update the production model
    # update_production_model(models[model_name])
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