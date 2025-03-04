from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from kaggle.api.kaggle_api_extended import KaggleApi
import os
from sklearn.model_selection import train_test_split

# Experiment setup
experiment_name = "debugging_experiment_CREATION_VIA_TRAIN.PY"
mlflow.set_experiment(experiment_name)
client = MlflowClient()

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
logger = logging.getLogger("predict")

app = FastAPI()

class PredictModelRequest(BaseModel):
    model_name: str = "RFC"
    dataset: str = "Mitbih"
    

@app.get("/status")
async def get_status():
    return {"status": "Prediction API is up"}

@app.post("/predict")
async def make_prediction(request: PredictModelRequest):
    model_name = request.model_name
    dataset = request.dataset
    model_name = model_name + "_" + dataset
    logger.info(f"Received prediction_realtime request with model: {model_name}")

    try:
        ml_model_deployed = load_deployment_model(model_name=model_name)
        logger.info("ml_Model loaded successfully with our load_deployment_model function")

        dataset_name = model_name.split("_")[-1] + "_test"
        if dataset_name not in datasets:
            logger.error(f"Dataset {dataset_name} not found in datasets dictionary")
            return {"error": f"Dataset {dataset_name} not found in datasets dictionary"}

        data_path = "../data/"
        download_datasets(data_path)
        logger.info(f"Datasets downloaded to {data_path}")
      
        dataset_path = "../data/heartbeat/"
        cached_datasets = prepare_datasets(dataset_path)
        logger.info(f"Datasets prepared from {dataset_path}")

        X_test = cached_datasets[f"X_test_{model_name.split('_')[-1]}"]
        y_test = cached_datasets[f"y_test_{model_name.split('_')[-1]}"]

        rand_row, rand_target = select_random_row(X_test=X_test, y_test=y_test)
        logger.info("Random row selected from test data")

        with mlflow.start_run():
            logger.info("mlflow.start_run() entered")
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("dataset_name", dataset_name)

            if isinstance(rand_row, pd.Series):
                rand_row = rand_row.values.reshape(1, -1)
            elif isinstance(rand_row, np.ndarray):
                rand_row = rand_row.reshape(1, -1)
            logger.info("rand_row successfully prepared and beginning to predict.") # Rand row debug print:", rand_row # --> Not useful for debugging the entire row... also logger.info is not used like print() syntax-wise!
            prediction = predict_with_ml_model(ml_model=ml_model_deployed, X=rand_row)
            logger.info("predictions with ML-Model from deployment made successfully")
            
            prediction_result = {
                "prediction": prediction.tolist() if isinstance(prediction, np.ndarray) else prediction
            }

            mlflow.log_param("input_data", rand_row.tolist() if isinstance(rand_row, np.ndarray) else rand_row.to_dict())
            mlflow.log_param("true_value", rand_target.tolist() if isinstance(rand_target, np.ndarray) else rand_target)

            if isinstance(prediction_result["prediction"], list):
                mlflow.log_param("predicted_value", prediction_result["prediction"][0])
            else:
                mlflow.log_param("predicted_value", prediction_result["prediction"])

        logger.info(f"Prediction successful: {prediction_result}")
        logger.info(f"True value: {rand_target}")
        logger.info("-------------------------------------------------------------------------------------------------------------")
        return {"prediction": prediction_result, "true_value": int(rand_target)}

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return {"Error during prediction": str(e)}
    
def load_deployment_model(model_name):
    model_uri = f"models:/{model_name}@deployment"
    logger.info(f"model_uri from load_deployment_model: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)
    return model

def select_random_row(X_test, y_test):
    random_index = np.random.randint(0, len(X_test))
    rand_row = pd.DataFrame(X_test.iloc[random_index] if hasattr(X_test, 'iloc') else X_test[random_index]).T
    rand_target = y_test.iloc[random_index] if hasattr(y_test, 'iloc') else y_test[random_index]
    logger.info(f"random index for random row: {random_index}")
    return rand_row, rand_target

def download_datasets(download_path, dataset_owner="shayanfazeli", dataset_name="heartbeat"):
    api = KaggleApi()
    api.authenticate()

    dataset_folder = os.path.join(download_path, dataset_name)
    if not os.path.exists(dataset_folder):
        api.dataset_download_files(dataset_owner + "/" + dataset_name, path=dataset_folder, unzip=True)
        logger.info("Datasets are downloaded and unzipped.")
    else:
        missing_files = [] 
        for file_name in ["mitbih_test.csv", "mitbih_train.csv", "ptbdb_abnormal.csv", "ptbdb_normal.csv"]:  
            file_path = os.path.join(dataset_folder, file_name)
            if not os.path.exists(file_path):
                missing_files.append(file_name)

        if missing_files:
            api.dataset_download_files(dataset_owner + "/" + dataset_name, path=dataset_folder, unzip=True, force=True)
            logger.info("Missing data was downloaded and unzipped. All Datasets are now available.")
        else:
            logger.info("All Datasets are already available.")

dataset_cache = {}

def prepare_datasets(path_to_dataset):
    global dataset_cache
    if path_to_dataset in dataset_cache:
        logger.info("Using cached datasets")
        return dataset_cache[path_to_dataset]

    mitbih_test, mitbih_train, ptbdb_abnormal, ptbdb_normal = load_datasets_in_workingspace(path_to_datasets=path_to_dataset)
    
    ptbdb_concated = pd.concat([ptbdb_abnormal, ptbdb_normal], ignore_index=True).sample(frac=1, random_state=42)
    X_ptbdb = ptbdb_concated.iloc[:, :-1]
    y_ptbdb = ptbdb_concated.iloc[:, -1]
    X_train_ptbdb, X_test_ptbdb, y_train_ptbdb, y_test_ptbdb = train_test_split(X_ptbdb, y_ptbdb, test_size=0.25, random_state=42)
    
    mitbih_concated = pd.concat([mitbih_test, mitbih_train], ignore_index=True).sample(frac=1, random_state=42)
    X_mitbih = mitbih_concated.iloc[:, :-1]
    y_mitbih = mitbih_concated.iloc[:, -1]
    X_train_mitbih, X_test_mitbih, y_train_mitbih, y_test_mitbih = train_test_split(X_mitbih, y_mitbih, test_size=0.25, random_state=42)

    logger.info("All test and train sets successfully prepared.")

    dataset_cache[path_to_dataset] = {
        "X_train_Ptbdb": X_train_ptbdb,
        "X_test_Ptbdb": X_test_ptbdb,
        "y_train_Ptbdb": y_train_ptbdb,
        "y_test_Ptbdb": y_test_ptbdb,
        "X_train_Mitbih": X_train_mitbih,
        "X_test_Mitbih": X_test_mitbih,
        "y_train_Mitbih": y_train_mitbih,
        "y_test_Mitbih": y_test_mitbih
    }

    return dataset_cache[path_to_dataset]

def load_datasets_in_workingspace(path_to_datasets="./heartbeat"):
    mitbih_test = pd.read_csv(path_to_datasets + "/" + "mitbih_test.csv", header=None)
    mitbih_train = pd.read_csv(path_to_datasets + "/" + "mitbih_train.csv", header=None)
    ptbdb_abnormal = pd.read_csv(path_to_datasets + "/" + "ptbdb_abnormal.csv", header=None)
    ptbdb_normal = pd.read_csv(path_to_datasets + "/" + "ptbdb_normal.csv", header=None)
    return mitbih_test, mitbih_train, ptbdb_abnormal, ptbdb_normal

def predict_with_ml_model(ml_model, X):
    logger.info("Debug: Entered the predict_with_ml_model function")
    try: 
        predictions = ml_model.predict(X)
        logger.info("Predictions made successfully.")
        return predictions
    except Exception as e:
        logger.info(f"Error making predictions in the predict_with_ml_model_function: {e}")
        return None

datasets = {
    "Mitbih_test": "../data/heartbeat/mitbih_test.csv",
    "Mitbih_train": "../data/heartbeat/mitbih_train.csv",
    "Ptbdb_test": "../data/heartbeat/ptbdb_test.csv",
    "Ptbdb_train": "../data/heartbeat/ptbdb_train.csv"
}
