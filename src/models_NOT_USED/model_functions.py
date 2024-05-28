import joblib
import numpy as np
#import tensorflow as tf #Version 2.13.0 is required since this was used by Kaggle to produce the .weights.h5 files --> Outsource tf for DL models into completely different container.
#PUT THIS INTO REQUIREMENTS.TXT --> Tensorflow MUST be 2.13.0!!! We don´t really need to import tensorflow, but it must be installed as version 2.13.0

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


def load_ml_model(path_to_model):
    logging.info("Entered the load_ml_models_function")
    """
    Load a machine learning model from a .pkl file.

    Parameters:
    path_to_model (str): Path to the .pkl file containing the model.

    Returns:
    model: The loaded machine learning model.
    """
    try:
        model = joblib.load(path_to_model)
        print(f"Model loaded successfully from {path_to_model}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_with_ml_model(ml_model, X):
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
        print("Predictions made successfully.")
        return predictions
    except Exception as e:
        print(f"Error making predictions in the predict_with_ml_model_function: {e}")
        return None