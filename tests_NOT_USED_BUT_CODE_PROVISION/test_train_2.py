#### WARNING: THE CORRECT EXECUTION OF THIS SCRIPT IS ONLY DOABLE SO: Python -m pytest test_train_2.py!!! THEN MLFLOW IS IMPORTED CORRECTLY!!! #############
import pytest
from fastapi.testclient import TestClient
from src.train.train import app
from unittest.mock import patch, MagicMock
import warnings
import logging

warnings.filterwarnings("ignore", category=DeprecationWarning) #this makes the deprecation warnings disappear, because they cannot be solved easily and require a migration of different packages. HOWEVER, they do not disturb the functioning of the code!

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

# Utility to capture log output
class LogCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)

    def get_logs(self):
        return [record.getMessage() for record in self.records]

@pytest.fixture
def log_capture():
    log_capture = LogCapture()
    logger = logging.getLogger("src.train.train")
    logger.addHandler(log_capture)
    yield log_capture
    logger.removeHandler(log_capture)

def test_train_model_success(mock_kaggle_api, mock_download_datasets, mock_prepare_datasets, mock_mlflow, log_capture):
    request_data = {
        "model_name": "RFC",
        "dataset": "Mitbih",
        "model_params": {"n_estimators": 10}
    }

    response = client.post("/train", json=request_data)
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "trained"

    logs = log_capture.get_logs()
    print("logs from function test_train_model_success:")
    print(logs)
    assert any("------------------------------Model training successful---------------------------------" in log for log in logs), "Model training log not found"

def test_train_model_dataset_not_found(mock_kaggle_api, mock_download_datasets, mock_prepare_datasets, mock_mlflow, log_capture):
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

    logs = log_capture.get_logs()
    assert any("Dataset not found" in log for log in logs), "Dataset not found log not found"

def test_get_status():
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == {"status": "Train API is up"}
