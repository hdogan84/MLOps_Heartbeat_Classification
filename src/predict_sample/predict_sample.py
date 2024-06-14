### This is the endpoint that is more or less the same as the predict endpoint, but does not create a random but receive it --> There needs to be another function / endpoint (maybe in this file) that continuesly calls this endpoint
# two more endpoints:
# 1) predict_simulation: Just like predict.py, but receives random row and does not create it
# 2) call_predict_simulation: uses code from predict.py, to create a random row and post it 60 times per minute to /predict_simulation

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import logging
import requests
import json
from pathlib import Path
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from kaggle.api.kaggle_api_extended import KaggleApi
import os
import httpx
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
logger = logging.getLogger("predict_sample")

app = FastAPI()

class PredictModelRequest(BaseModel):
    model_name: str = "RFC"
    dataset: str = "Mitbih"
    x_sample: List = 187 * [0.1]
    

@app.get("/status")
async def get_status():
    return {"status": "Prediction API is up"}

    
@app.post("/predict_sample")
async def make_prediction(request: PredictModelRequest):
    # Modified the Pydantic model to add one sample of feature data.
    # Should later add to the Pydantic Model the Label parameter (Y = x.iloc[i,187])
    model_name = request.model_name
    dataset = request.dataset
    x_sample = request.x_sample # This corresponds to one row of data, without the label

    model_name = model_name + "_" + dataset
    logger.info(f"Received prediction_sample request with model: {model_name}")

    try:
        ml_model_deployed = load_deployment_model(model_name=model_name)
        logger.info("ml_Model loaded successfully with our load_deployment_model function")


        with mlflow.start_run():
            logger.info("mlflow.start_run() entered")
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("dataset_name", dataset)            

            # This is the input feature variable
            rand_row = np.array(x_sample).reshape(1,-1)

            logger.info("Beginning to predict.") # Rand row debug print:", rand_row # --> Not useful for debugging the entire row... also logger.info is not used like print() syntax-wise!
            prediction = predict_with_ml_model(ml_model=ml_model_deployed, X=rand_row)
            logger.info("predictions with ML-Model from deployment made successfully")
            logger.info(f"Type of prediction is {type(prediction)}")

            prediction_result = {
                "prediction": prediction.tolist() if isinstance(prediction, np.ndarray) else prediction
            }

            mlflow.log_param("input_data", rand_row.tolist() if isinstance(rand_row, np.ndarray) else rand_row.to_dict())
            
            if isinstance(prediction_result["prediction"], list):
                mlflow.log_param("predicted_value", prediction_result["prediction"][0])
            else:
                mlflow.log_param("predicted_value", prediction_result["prediction"])

        logger.info(f"Prediction successful: {prediction_result}")

        logger.info("-------------------------------------------------------------------------------------------------------------")
        
        return prediction_result

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return {"Error during prediction": str(e)}

    
def load_deployment_model(model_name):
    model_uri = f"models:/{model_name}@deployment"
    logger.info(f"model_uri from load_deployment_model: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)
    return model

def predict_with_ml_model(ml_model, X):
    logger.info("Debug: Entered the predict_with_ml_model function")
    try: 
        predictions = ml_model.predict(X)
        logger.info("Predictions made successfully.")
        return predictions
    except Exception as e:
        logger.info(f"Error making predictions in the predict_with_ml_model_function: {e}")
        return None


