import joblib
import numpy as np
import tensorflow as tf #Version 2.13.0 is required since this was used by Kaggle to produce the .weights.h5 files
#PUT THIS INTO REQUIREMENTS.TXT --> Tensorflow MUST be 2.13.0!!! We don´t really need to import tensorflow, but it must be installed as version 2.13.0

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
    

def load_advanced_cnn_model(model_path, num_classes=5):
    """
    builds the advanced CNN model from the reports
    model_path = path where the model is stored
    num_classes = number of classes for the target on which the model was trained (and on which is predicted) --> 5 for Mitbih, 2 for Ptbdb
    """

    class Config_Advanced_CNN:
        Conv1_filter_num = 32
        Conv1_filter_size = 3
        

    adv_cnn_model = tf.keras.models.Sequential()
    adv_cnn_model.add(tf.keras.layers.Conv1D(Config_Advanced_CNN.Conv1_filter_num, Config_Advanced_CNN.Conv1_filter_size, activation='relu', input_shape=(187, 1))) # We add one Conv1D layer to the model
    adv_cnn_model.add(tf.keras.layers.MaxPooling1D(pool_size=3, strides=2)) # We add one Conv1D layer to the model
    adv_cnn_model.add(tf.keras.layers.Conv1D(Config_Advanced_CNN.Conv1_filter_num//2, Config_Advanced_CNN.Conv1_filter_size, activation='relu' )) # We add one Conv1D layer to the model
    adv_cnn_model.add(tf.keras.layers.MaxPooling1D(pool_size=3, strides=2)) # We add one Conv1D layer to the model
    adv_cnn_model.add(tf.keras.layers.Flatten()) # After  
    adv_cnn_model.add(tf.keras.layers.Dropout(rate=0.2))
    adv_cnn_model.add(tf.keras.layers.Dense(120, activation=tf.keras.layers.LeakyReLU(alpha=0.001)))
    adv_cnn_model.add(tf.keras.layers.Dense(60, activation=tf.keras.layers.LeakyReLU(alpha=0.001)))
    adv_cnn_model.add(tf.keras.layers.Dense(20, activation=tf.keras.layers.LeakyReLU(alpha=0.001)))
    adv_cnn_model.add(tf.keras.layers.Dense(num_classes, activation='softmax')) #softmax classes are dynamically adjusted according to the dataset!
    
    adv_cnn_model.load_weights(model_path)
    return adv_cnn_model

#### This function is essentially the same as the predict_with_ml_model function --> One function for all models and types of prediction later?
## --> Nope, dl models need the .argmax(axis=1) argument to select the correct class.
def predict_with_dl_model(dl_model, X):
    """
    Predict using a loaded deep learning model.

    Parameters:
    ml_model: The loaded machine learning model.
    X (array-like): The input data for prediction.

    Returns:
    array-like: The predictions made by the model.
    """
    try:
        predictions = dl_model.predict(X).argmax(axis=1)
        print("Predictions made successfully.")
        return predictions
    except Exception as e:
        print(f"Error making predictions: {e}")
        return None