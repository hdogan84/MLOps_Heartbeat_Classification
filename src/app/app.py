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
from models.model_functions import load_ml_model, predict_with_ml_model
from app_functions import select_random_row
import numpy as np
import pandas as pd

## Side note: If you cannot import the selfwritten modules, this might help, especially when working with venv: https://stackoverflow.com/questions/71754064/vs-code-pylance-problem-with-module-imports


from typing import List, Dict
import pandas as pd
import json
from pydantic import BaseModel
import uvicorn

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

##### OLD CODE BELOW (IS WORKING FINE WITHOUT AUTHENTIFICATION) ##########

## --> NOW THE DEPEND ROUTES HAVE TO BE CONFIGURED IN THE SINGLE ENDPOINTS OR AM I MISSING SOMETHING? ####

# Placeholder model database
models = {
    "model_v1": "path/to/model_v1",
    "model_v2": "path/to/model_v2", 
    "RFC_Mitbih_gridsearch": "../models/ML_Models/RFC_Optimized_Model_with_Gridsearch_MITBIH_A_Original.pkl"
}

# Placeholder for metrics storage
model_metrics = {
    "model_v1": {"accuracy": 0.95, "confusion_matrix": [[50, 2], [1, 47]]},
    "model_v2": {"accuracy": 0.96, "confusion_matrix": [[51, 1], [2, 46]]}
}

# Placeholder for dataset names (and links) --> Make this with environment variables or some other type of variable instead of hard-coding
datasets = {
    "Mitbih_test": "../data/heartbeat/mitbih_test.csv",
    "Mitbih_train": "../data/heartbeat/mitbih_train.csv",

    "Ptbdb_test": "../data/heartbeat/ptbdb_test.csv",
    "Ptbdb_train": "../data/heartbeat/ptbdb_train.csv"
}

# Placeholder for notifications
notifications = []

# Endpoint to check API status
@app.get("/status")
async def get_status():
    return {"status": 1}

# Endpoint to predict an EKG signal in real-time
class EKGSignal(BaseModel):
    signal: List[float]

@app.post("/predict_realtime")
async def predict_realtime(ekg_signal: EKGSignal, model_name: str = "RFC_Mitbih_gridsearch", dataset_name: str = "Mitbih_test"):
    if model_name not in models:
        return {"error": "Model not found"}
    if dataset_name not in datasets:
        return {"error": "Dataset not found"}
    
    #load datasets if not already happened (this has to be checked in each function!)
    data_path = "../data/"
    download_datasets(data_path)
    #make the test and train sets
    dataset_path = "../data/heartbeat/"

    #Generation of the train and test variables --> Takes a lot of time and should be made available globally if possible? Also checked if already available. 
    X_train_ptbdb, X_test_ptbdb, y_train_ptbdb, y_test_ptbdb, X_train_mitbih, X_test_mitbih, y_train_mitbih, y_test_mitbih = prepare_datasets(dataset_path)
    
    # Load model and make prediction --> We start with a simple ML .pkl model --> Later a distinction must be made in the models dictionary?
    ml_model = load_ml_model(models[model_name])

    #select a random row from the selected dataset
    # If structure needed to load the correct dataset only
    rand_row_mitbih, rand_target_mitbih = select_random_row(X_test=X_test_mitbih, y_test=y_test_mitbih) 

    #make the prediction with the selected random row
    #distinquish between the ML and DL models with if structure for prediction
    prediction = predict_with_ml_model(ml_model=ml_model, X=rand_row_mitbih)
    # Ensure the prediction is JSON serializable
    prediction_result = {
        "prediction": prediction.tolist() if isinstance(prediction, np.ndarray) else prediction#,
        #"true_value": rand_target_mitbih #do not show the true value because we want to simulate a livesession where the true value is unknown
    }

    print("prediction value:", prediction_result)
    print("true value:", rand_target_mitbih)

    return prediction_result

"""# Endpoint to predict on a batch dataset and return metrics
@app.post("/predict_batch")
async def predict_batch(dataset: str, model_name: str):
    if model_name not in models:
        return {"error": "Model not found"}
    
    # Load dataset and model, perform batch prediction, compute metrics
    # data = pd.read_csv(dataset)
    # model = load_model(models[model_name])
    # metrics = evaluate_model(model, data)
    metrics = model_metrics[model_name]  # Placeholder
    return metrics"""

# Endpoint to retrain a model on a new dataset
@app.post("/retrain")
async def retrain_model(dataset: str, model_name: str):
    if model_name not in models:
        return {"error": "Model not found"}
    
    # Load dataset, model, and perform retraining
    # data = pd.read_csv(dataset)
    # model = load_model(models[model_name])
    # new_model, new_metrics = retrain(model, data)
    
    # Save the new model and log metrics (dummy response here)
    new_model_name = model_name + "_retrained"
    models[new_model_name] = "path/to/new_model"
    model_metrics[new_model_name] = {"accuracy": 0.97, "confusion_matrix": [[52, 0], [1, 47]]}
    
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