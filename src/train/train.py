from fastapi import FastAPI
import logging
from pathlib import Path
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, ConfigDict
import os
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier as RFC

# Define the path for the log file. This code is the same in each container, the log file is a bind mount defined in the docker-compose.yaml --> All containers write in this bind mount log file.
log_file_path = Path("reports/logs/app.log")

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


class TrainModelRequest(BaseModel):
    model_name: str = "RFC"
    dataset: str = "Mitbih"
    model_params: dict = {}

    model_config = ConfigDict(
        protected_namespaces=()
    )

@app.get("/status")
async def get_status():
    return {"status": "Train API is up"}

@app.post("/train")
async def train_model(request: TrainModelRequest):
    model_name = request.model_name
    dataset = request.dataset
    model_params = request.model_params

    if "RFC" in model_name:
        model = RFC(**model_params)
    logging.info(f"Initiated {model_name} trainer")

    dataset_name = f"{dataset}_train"

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

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        metrics = {"accuracy": acc}

        new_model_name = model_name + "_" + dataset

        # Ensure the experiment exists --> This experiment name is set on startup, even if the endpoint is not triggered! Not the biggest bug, but keep an eye open.
        experiment_name = "debugging_experiment_CREATION_VIA_TRAIN.PY" #this could be passed as argument or the whole structure could be used in gateway-api or completely different endpoint. Must be included in update.py as well?
        mlflow.set_experiment(experiment_name)
        
        # Initialize the client after setting the experiment
        client = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            logging.info(f"Experiment {experiment_name} does not exist. Creating a new one.")
            experiment_id = client.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id

        with mlflow.start_run(experiment_id=experiment_id):
            mlflow.log_params({"dataset": dataset, "model_name": new_model_name})
            mlflow.log_metrics(metrics)
            relative_model_path = os.path.relpath(new_model_name, start=os.getcwd())
            logging.info(f"relative_model_path from register_model(): {relative_model_path}")
            mlflow.sklearn.log_model(model, artifact_path=relative_model_path, registered_model_name=new_model_name)

        logging.info("------------------------------Model training successful---------------------------------")
        return {"status": "trained", "model_name": new_model_name, "metrics": metrics}

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
    logging.info("Entered the download_datasets function.")
    api = KaggleApi()
    api.authenticate()

    dataset_folder = os.path.join(download_path, dataset_name)
    logging.info(f"dataset_folder from download_datasets(): {dataset_folder}")
    if not os.path.exists(dataset_folder):
        api.dataset_download_files(dataset_owner + "/" + dataset_name, path=dataset_folder, unzip=True)
        logging.info("Datasets are downloaded and unzipped.")
    else:
        missing_files = []
        for file_name in ["mitbih_test.csv", "mitbih_train.csv", "ptbdb_abnormal.csv", "ptbdb_normal.csv"]:
            file_path = os.path.join(dataset_folder, file_name)
            if not os.path.exists(file_path):
                missing_files.append(file_name)

        if missing_files:
            api.dataset_download_files(dataset_owner + "/" + dataset_name, path=dataset_folder, unzip=True, force=True)
            logging.info("Missing data was downloaded and unzipped. All Datasets are now available.")
        else:
            logging.info("All Datasets are already available.")

global dataset_cache
dataset_cache = {}
def prepare_datasets(path_to_dataset):
    global dataset_cache
    if path_to_dataset in dataset_cache:
        print("Using cached datasets")
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

    print("All test and train sets successfully prepared.")

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
