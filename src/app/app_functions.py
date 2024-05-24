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





import numpy as np
import pandas as pd

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


### Functions for mlflow #######


# load_pickle_model: Already implemented in the model functions

import mlflow
import mlflow.sklearn
import os
#register the pickle model as an MLFlow model
def register_model(model, model_name):
    #with mlflow.start_run() as run: #we do not need to start a separate run, because this function already runs in a mlflow run?
    
    relative_model_path = os.path.relpath(model_name, start=os.getcwd()) #fetching the relative path for the model_name
    logging.info(f"relative_model_path from register_model(): {relative_model_path}")
    mlflow.sklearn.log_model(model, artifact_path=relative_model_path, registered_model_name=model_name) #this should produce a relative path in meta.yaml for the registered (version of the) model.
    #mlflow.sklearn.log_model(model, artifact_path=model_name, registered_model_name=model_name) #this produces absolute paths which will not work in docker or on other computers
    accuracy = 0.7 #this is a dummy variable, the accuracy should come from a evaulaute_function.
    mlflow.log_metric("accuracy", accuracy) #where is this metric stored? Is identifiable because so long...
    logging.info(f"Model registered with name: {model_name}")

# train and register a new version of the model
# Assumption: The train_model function is implemented (is not!) --> Dummy and not usable
# --> Hakan wrote this code for the ML_Models, we can maybe use his version
# and just make the name for the classifier simpler, like "RFC"
def train_and_register_new_version(model_name, train_data, train_labels, new_params):
    model = train_model(train_data, train_labels, new_params)
    register_model(model, model_name)
    print(f"New version of model {model_name} registered")


# Find the best model and set alias to "deployment"
#This requires the model name to be more simple, like "RFC" --> Changed in the dictionary
from mlflow.tracking import MlflowClient

def set_deployment_alias(model_name, metric_name):
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    
    best_version = None
    best_metric = float('-inf')
    logging.info(f"Versions found in set_deployment_alias: {versions}")

    for version in versions:
        run_id = version.run_id
        logging.info(f"run_id from function set_deployment_alias: {run_id}")
        run = client.get_run(run_id)
        logging.info(f"run from client.get_run() inside set_deployment_alias(): {run}")
        metrics = run.data.metrics
        logging.info(f"metrics from run.data.metrics inside set_deployment_alias: {metrics}")

        # Check if the desired metric is available
        if metric_name in metrics:
            if metrics[metric_name] > best_metric:
                best_metric = metrics[metric_name]
                best_version = version.version

    if best_version:
        logging.info(f"Best Version found from function set_deployment alias: {best_version}")
        
        # First, remove the "deployment" alias from all versions
        for version in versions:
            client.delete_registered_model_alias(model_name, alias="deployment")
            client.delete_registered_model_alias(model_name, alias="not_deployment")
            client.delete_registered_model_alias(model_name, alias=f"not_deployment_{version.version}")
        
        # Set "not_deployment" alias for all versions first
        for version in versions:
            if version.version != best_version:
                client.set_registered_model_alias(model_name, f"not_deployment_{version.version}", version.version)
                logging.info(f"Set 'not_deployment' alias for version {version.version}")
        
        # Set "deployment" alias for the best version
        client.set_registered_model_alias(model_name, "deployment", best_version)
        logging.info(f"Set version {best_version} as deployment for model {model_name}")
    else:
        logging.info("No suitable model found to set as deployment")
# load the deployment model
# --> Usable in the /predict_realtime endpoint
#BIG Question: Is the model_uri path working this time?

def load_deployment_model(model_name):
    model_uri = f"models:/{model_name}@deployment"
    logging.info(f"model_uri from load_deployment_model: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)
    return model

# Predict with deployment model
# is already implemented as predict_with_ml_model in the model_functions.py file



