#### WARNING: THE CORRECT EXECUTION OF THIS SCRIPT IS ONLY DOABLE SO: Python -m pytest test_train_2.py!!! THEN MLFLOW IS IMPORTED CORRECTLY!!! #############
##### THE TESTS IN THIS SCRIPT ONLY CHECK THE FUNCTIONING OF THE ENDPOINTS BUT CANNOT CORRECTLY TEST THAT A MODEL IS TRAINED CORRECTLY!!!!!!!!!!#############
        #### --> This would be enough, if we would have our models as .pkl file stored locally?
        ##### --> This file could be made even easier and just call the endpoints with the mocked dataset to test if a model is trained (but then subsequently the mlflow registry should be checked for a newly added model version? This might be overkill, if the correct response is given back, everything is ok)
        #### --> This requires this file beeing called from github workflows yaml inside a docker container or with the docker compose run pytest [...] command.

import pytest
from fastapi.testclient import TestClient
#from src.train.train import app #this is only valid if the train.py is only available in sr/train/train.py 
from train import app #this is valid for the github actions, if the test_train.py and train.py are in the same folder (container)
##--> Produces import error on github, but in the example, it is the same syntax!!: https://github.com/DataScientest-Studio/juin23_continu_mlops_pompiers/tree/master/src/api_user
from unittest.mock import patch, MagicMock
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) #this makes the depreceation warnings disappear, because they cannot be solved easily and require an migration of different packages. HOWEVER, they do not disturb the functioning of the code!

client = TestClient(app)

@pytest.fixture
def mock_datasets():
    return {
        "X_train_Mitbih": [[0.1, 0.2, 0.3]] * 10,
        "y_train_Mitbih": [0, 1] * 5,
        "X_test_Mitbih": [[0.1, 0.2, 0.3]] * 4,
        "y_test_Mitbih": [0, 1] * 2
    }

@pytest.fixture
def mock_mlflow():
    with patch("src.train.train.mlflow") as mock_mlflow:
        yield mock_mlflow

@pytest.fixture
def mock_kaggle_api():
    with patch("src.train.train.KaggleApi") as mock_kaggle:
        mock_kaggle_instance = MagicMock()
        mock_kaggle.return_value = mock_kaggle_instance
        mock_kaggle_instance.authenticate.return_value = None
        yield mock_kaggle_instance

@pytest.fixture
def mock_download_datasets():
    with patch("src.train.train.download_datasets") as mock_download:
        yield mock_download

@pytest.fixture
def mock_prepare_datasets(mock_datasets):
    with patch("src.train.train.prepare_datasets") as mock_prepare:
        mock_prepare.return_value = mock_datasets
        yield mock_prepare

def test_train_model_success(mock_kaggle_api, mock_download_datasets, mock_prepare_datasets, mock_mlflow):
    request_data = {
        "model_name": "RFC",
        "dataset": "Mitbih",
        "model_params": {"n_estimators": 10}
    }

    response = client.post("/train", json=request_data)
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "trained"

def test_train_model_dataset_not_found(mock_kaggle_api, mock_download_datasets, mock_prepare_datasets, mock_mlflow):
    mock_prepare_datasets.side_effect = Exception("Dataset not found")

    request_data = {
        "model_name": "RFC",
        "dataset": "NonExistentDataset",
        "model_params": {"n_estimators": 10}
    }

    response = client.post("/train", json=request_data)
    assert response.status_code == 200
    assert "error" in response.json()
    assert response.json()["error"] == "Dataset not found"

def test_get_status():
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == {"status": "Train API is up"}
