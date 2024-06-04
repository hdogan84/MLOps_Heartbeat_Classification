from fastapi import FastAPI
import logging
from pathlib import Path
from mlflow.tracking import MlflowClient
from pydantic import BaseModel #needed for making the request via gateway api possible?

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

class UpdateModelRequest(BaseModel): #the basemodel to make the request via gateway api possible?
    model_name: str = "RFC"
    dataset: str = "Ptbdb"
    metric_name: str = "accuracy"


@app.get("/status")
async def get_status():
    return {"status": "Update API is up"}


@app.post("/update")
async def update_deployment_model(request: UpdateModelRequest): # no instantiation here
    """
    The actual updating code from backup_codes. copied from set_deployment_alias out of the backup functions.
    """
    try:
        model_name = request.model_name  # Extracting values from the request object
        dataset = request.dataset
        metric_name = request.metric_name

        ## assigning the model name to include the dataset name
        model_name = model_name + "_" + dataset

        client = MlflowClient() #the mlfow tracking uri is set as environment variable, therefore this should work for now.
        versions = client.search_model_versions(f"name='{model_name}'")
        if versions:
            pass
        else:
            logging.error("Error during updating: model / dataset combination could not be found in the MLFlow model registry")
            return {"Error during updating": "model / dataset combination could not be found in the MLFlow model registry"}

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
            else:
                logging.error("Error during updating: metric_name is not defined / available")
                return {"Error during updating": "metric_name is not defined / available"}
                
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
        return {"deployment_model set to": model_name + " Version " + best_version}

    except Exception as e:
        logging.error(f"Error during updating: {e}")
        return {"Error during updating": str(e)}