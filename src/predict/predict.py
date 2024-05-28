from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
#V1: Tracking Uri is set as environment variable in docker-compose.yaml
#mlflow.set_tracking_uri("file:///app/mlruns")#V1 for Docker--> This must be path inside the container!
#mlflow.set_tracking_uri("./mlruns") #old code
experiment_name = "debugging_experiment" #this must later be definable from the gateway api?
mlflow.set_experiment(experiment_name)
from mlflow.tracking import MlflowClient
client = MlflowClient() #define the client after setting the tracking uri, otherwise a not used mlruns directory will be created in the app folder (undesirable)

from kaggle.api.kaggle_api_extended import KaggleApi
import os
from sklearn.model_selection import train_test_split
# Define the path for the log file. This code is the same in each container, the log file is a bind mount defined in the docker-compose.yaml --> All containers write in this bind mount log file.
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

app = FastAPI()

class PredictModelRequest(BaseModel): #the basemodel to make the request via gateway api possible?
    model_name: str = "RFC"
    dataset: str = "Mitbih"
    

@app.get("/status")
async def get_status():
    return {"status": "Prediction API is up"}


#this functions must be copied from the basic scripts.
@app.post("/predict")
async def make_prediction(request: PredictModelRequest):
    model_name = request.model_name
    dataset = request.dataset
    model_name = model_name + "_" + dataset #concating it
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

        with mlflow.start_run(): #this can be left out, because it justs starts a new run (although in the correct experiment) which is uncecessary
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
    


def load_deployment_model(model_name):
    model_uri = f"models:/{model_name}@deployment"
    logging.info(f"model_uri from load_deployment_model: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)
    return model

def select_random_row(X_test, y_test):
    """
    Select a random row from the test dataset and its corresponding target.

    Parameters:
    X_test (pd.DataFrame or np.array): Test features dataset.
    y_test (pd.Series or np.array): Test target dataset.

    Returns:
    tuple: A tuple containing the random row from X_test and its corresponding target from y_test.
    """
    # Ensure the random selection is reproducible
    #np.random.seed(42) #this leads to selection of the same "random" row each time...just for debugging
    
    # Select a random index
    random_index = np.random.randint(0, len(X_test))

    
    # Get the random row and its corresponding target
    rand_row = pd.DataFrame(X_test.iloc[random_index] if hasattr(X_test, 'iloc') else X_test[random_index]).T #transformation is necessary? Is this bullshit and an upstream problem?
    #print("rand_row from app functions:", rand_row)
    rand_target = y_test.iloc[random_index] if hasattr(y_test, 'iloc') else y_test[random_index]
    logging.info(f"random index for random row: {random_index}")
    return rand_row, rand_target



def download_datasets(download_path, dataset_owner="shayanfazeli", dataset_name="heartbeat"):
    # Configure and authenticate with the Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Check if the dataset folder already exists
    dataset_folder = os.path.join(download_path, dataset_name)
    if not os.path.exists(dataset_folder):
        # Dataset folder does not exist --> Download and save the datasets
        api.dataset_download_files(dataset_owner + "/" + dataset_name, path=dataset_folder, unzip=True)
        logging.info("Datasets are downloaded and unzipped.")
    else:
        # Dataset folder exists, but datasets might be missing
        missing_files = [] 
        for file_name in ["mitbih_test.csv", "mitbih_train.csv", "ptbdb_abnormal.csv", "ptbdb_normal.csv"]:  
            file_path = os.path.join(dataset_folder, file_name)
            if not os.path.exists(file_path):
                missing_files.append(file_name)

        if missing_files:
            # If missing files are present, download ALL files and overwrite the old folder.
            api.dataset_download_files(dataset_owner + "/" + dataset_name, path=dataset_folder, unzip=True, force=True)
            logging.info("Missing data was downloaded and unzipped. All Datasets are now available.")
        else:
            logging.info("All Datasets are already available.")


dataset_cache = {} #this must be defined here! Otherwise a new empty dataset_cache is introduced each time the function or endpoint is called.
def prepare_datasets(path_to_dataset):
    global dataset_cache
    # little check if dataset has been cached
    if path_to_dataset in dataset_cache:
        logging.info("Using cached datasets")
        return dataset_cache[path_to_dataset]

    #if the datasets with the specific path have not been generated / cached yet, do cache them (if later a feeding function is introduced, rework is necessary here.)
    # Load the datasets into the workspace
    mitbih_test, mitbih_train, ptbdb_abnormal, ptbdb_normal = load_datasets_in_workingspace(path_to_datasets=path_to_dataset)
    
    # Concatenate and shuffle ptbdb datasets
    ptbdb_concated = pd.concat([ptbdb_abnormal, ptbdb_normal], ignore_index=True).sample(frac=1, random_state=42)
    X_ptbdb = ptbdb_concated.iloc[:, :-1]
    y_ptbdb = ptbdb_concated.iloc[:, -1]  # assuming the last column is the label
    X_train_ptbdb, X_test_ptbdb, y_train_ptbdb, y_test_ptbdb = train_test_split(X_ptbdb, y_ptbdb, test_size=0.25, random_state=42)
    
    # Concatenate and shuffle mitbih datasets
    mitbih_concated = pd.concat([mitbih_test, mitbih_train], ignore_index=True).sample(frac=1, random_state=42)
    X_mitbih = mitbih_concated.iloc[:, :-1]
    y_mitbih = mitbih_concated.iloc[:, -1]  # assuming the last column is the label
    X_train_mitbih, X_test_mitbih, y_train_mitbih, y_test_mitbih = train_test_split(X_mitbih, y_mitbih, test_size=0.25, random_state=42)

    # Print success message
    logging.info("All test and train sets successfully prepared.")

   # Cache the datasets
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
    #reading in the datasets from the local ../data folder --> this folder is not pushed on github and only locally available.
    mitbih_test = pd.read_csv(path_to_datasets + "/" + "mitbih_test.csv",header=None)
    mitbih_train = pd.read_csv(path_to_datasets + "/" + "mitbih_train.csv",header=None)
    ptbdb_abnormal = pd.read_csv(path_to_datasets + "/" + "ptbdb_abnormal.csv",header=None)
    ptbdb_normal = pd.read_csv(path_to_datasets + "/" + "ptbdb_normal.csv",header=None)
    return mitbih_test, mitbih_train, ptbdb_abnormal, ptbdb_normal

def predict_with_ml_model(ml_model, X): #could maybe be replaced with the mlflow.sklearn.predict() function or at least inherit this function? Any advantages?
    logging.info("Debug: Entered the predict_with_ml_model function")
    """
    Predict using a loaded machine learning model.

    Parameters:
    ml_model: The loaded machine learning model.
    X (array-like): The input data for prediction.

    Returns:
    array-like: The predictions made by the model.
    """
    try: 
        predictions = ml_model.predict(X)
        logging.info("Predictions made successfully.")
        return predictions
    except Exception as e:
        logging.info(f"Error making predictions in the predict_with_ml_model_function: {e}")
        return None

############ Outsourced extra code #########
# Placeholder for dataset names (and links)  --> Should be a file that is growable, for now only hardcoding
datasets = {
    "Mitbih_test": "../data/heartbeat/mitbih_test.csv",
    "Mitbih_train": "../data/heartbeat/mitbih_train.csv",

    "Ptbdb_test": "../data/heartbeat/ptbdb_test.csv",
    "Ptbdb_train": "../data/heartbeat/ptbdb_train.csv"
}
