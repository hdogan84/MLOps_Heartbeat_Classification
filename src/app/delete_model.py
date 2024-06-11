import mlflow
from mlflow.tracking import MlflowClient
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def delete_model_version(model_name: str, version: str):
    client = MlflowClient()
    try:
        # Delete a specific version of the model
        client.delete_model_version(name=model_name, version=version)
        logging.info(f"Successfully deleted version {version} of model {model_name}.")
    except Exception as e:
        logging.error(f"Error deleting version {version} of model {model_name}: {e}")

def delete_all_versions_of_model(model_name: str):
    client = MlflowClient()
    try:
        # Fetch all versions of the model
        versions = client.get_latest_versions(name=model_name)
        for version in versions:
            client.delete_model_version(name=model_name, version=version.version)
            logging.info(f"Deleted version {version.version} of model {model_name}.")

        # Delete the registered model itself
        client.delete_registered_model(name=model_name)
        logging.info(f"Successfully deleted all versions and the registered model {model_name}.")
    except Exception as e:
        logging.error(f"Error deleting all versions of model {model_name}: {e}")

if __name__ == "__main__":
    model_name_to_delete = "your_model_name"
    version_to_delete = "1"  # Change this to the version you want to delete

    # Uncomment the function call you want to test
    # delete_model_version(model_name=model_name_to_delete, version=version_to_delete)
    # delete_all_versions_of_model(model_name=model_name_to_delete)
