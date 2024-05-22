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

#register the pickle model as an MLFlow model
def register_model(model, model_name):
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(model, artifact_path=model_name, registered_model_name=model_name)
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
        run = client.get_run(run_id)
        metrics = run.data.metrics
        if metrics[metric_name] > best_metric:
            best_metric = metrics[metric_name]
            best_version = version.version
    
    if best_version:
        client.set_registered_model_alias(model_name, "deployment", best_version)
        for version in versions:
            if version.version != best_version:
                client.delete_registered_model_alias(model_name, version.version, "deployment")
                client.set_registered_model_alias(model_name, "not_deployment", version.version)
        logging.info(f"Set version {best_version} as deployment for model {model_name}")
    else:
        logging.info("No suitable model found to set as deployment")


# load the deployment model
# --> Usable in the /predict_realtime endpoint
#BIG Question: Is the model_uri path working this time?

def load_deployment_model(model_name):
    model_uri = f"models:/{model_name}@deployment"
    model = mlflow.sklearn.load_model(model_uri)
    return model

# Predict with deployment model
# is already implemented as predict_with_ml_model in the model_functions.py file



