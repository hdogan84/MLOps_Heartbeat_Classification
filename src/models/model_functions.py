import joblib
import numpy as np

def load_ml_model(path_to_model):
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
        print(f"Error making predictions: {e}")
        return None