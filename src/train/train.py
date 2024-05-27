from fastapi import FastAPI
import logging
from pathlib import Path
import mlflow
import mlflow.sklearn
#V1: Tracking Uri is set as environment variable in docker-compose.yaml
#mlflow.set_tracking_uri("file:///app/mlruns")#V1 for Docker--> This must be path inside the container!
#mlflow.set_tracking_uri("./mlruns") #old code
experiment_name = "debugging_experiment" #this must later be definable from the gateway api?
mlflow.set_experiment(experiment_name)
from mlflow.tracking import MlflowClient
client = MlflowClient() #define the client after setting the tracking uri, otherwise a not used mlruns directory will be created in the app folder (undesirable)
from pydantic import BaseModel #needed for making the request via gateway api possible?

import os
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier as RFC

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

class TrainModelRequest(BaseModel): #the basemodel to make the request via gateway api possible?
    model_name: str = "RFC"
    dataset: str = "Mitbih"
    model_params: dict = {}  # New field for model parameters


@app.get("/status")
async def get_status():
    return {"status": "Train API is up"}


@app.post("/train")
async def train_model(request: TrainModelRequest): # no instantiation here
    """
    Endpoint to train and register new models or new versions of existing models.
    Creates a new instance of the specified model and trains it on the selected dataset.
    Then registration (logging) on the MLFlow Webserver is done, without automatic setting of the best version as deployment model for this model_name.

    dataset: the dataset_name as string
    model_name: the model_name as string

    Future work:
    - use params_dict as argument to refine the model_training (same model_name)
    """
    model_name = request.model_name  # Extracting values from the request object
    dataset = request.dataset
    model_params = request.model_params
    
    if "RFC" in model_name: #Simon: This is more inclusive
        model = RFC(**model_params) #simon: Here some params in form of a dict could be passed.
    logging.info(f"Initiated {model_name} trainer") #simon: and the params could be logged.

    dataset_name = f"{dataset}_train" #adding the _train to the dataset argument

    if dataset_name not in datasets:
        logging.error(f"Dataset {dataset_name} not found")
        return {"error": "Dataset not found"}

    try:
        logging.info("Entered the try structure in train.py")
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
    
    
    try:
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
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        return {"error": "File not found"}
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return {"error": "Unexpected error"}






########## Some extra code / variables that are hardcoded / helper functions #######
datasets = {
    "Mitbih_test": "../data/heartbeat/mitbih_test.csv",
    "Mitbih_train": "../data/heartbeat/mitbih_train.csv",

    "Ptbdb_test": "../data/heartbeat/ptbdb_test.csv",
    "Ptbdb_train": "../data/heartbeat/ptbdb_train.csv"
}

def download_datasets(download_path, dataset_owner="shayanfazeli", dataset_name="heartbeat"):
    # Configure and authenticate with the Kaggle API
    logging.info("Entered the download_datasets function.")
    api = KaggleApi()
    api.authenticate()

    # Check if the dataset folder already exists
    dataset_folder = os.path.join(download_path, dataset_name)
    logging.info(f"dataset_folder from download_datasets(): {dataset_folder}")
    if not os.path.exists(dataset_folder):
        # Dataset folder does not exist --> Download and save the datasets
        api.dataset_download_files(dataset_owner + "/" + dataset_name, path=dataset_folder, unzip=True)
        print("Datasets are downloaded and unzipped.")
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
            print("Missing data was downloaded and unzipped. All Datasets are now available.")
        else:
            print("All Datasets are already available.")

global dataset_cache
dataset_cache = {} #this must be defined here! Otherwise a new empty dataset_cache is introduced each time the function or endpoint is called.
def prepare_datasets(path_to_dataset):
    global dataset_cache
    # little check if dataset has been cached
    if path_to_dataset in dataset_cache:
        print("Using cached datasets")
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
    print("All test and train sets successfully prepared.")

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
