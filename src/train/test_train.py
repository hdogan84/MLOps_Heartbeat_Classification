import pytest
from fastapi.testclient import TestClient
from train import app
from unittest.mock import patch
import logging

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
def mock_prepare_datasets(mock_datasets):
    with patch("train.prepare_datasets") as mock_prepare:
        mock_prepare.return_value = mock_datasets
        yield mock_prepare

def test_train_model_success(caplog):
    request_data = {
        "model_name": "RFC",
        "dataset": "Ptbdb",
        "model_params": {"n_estimators": 10}
    }

    with caplog.at_level(logging.INFO):
        response = client.post("/train", json=request_data)

    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "trained"

    print("logs from function test_train_model_success:")
    print([record.message for record in caplog.records])

    assert any("------------------------------Model training successful---------------------------------" in record.message for record in caplog.records), "Model training log not found"

def test_train_model_dataset_not_found(mock_prepare_datasets, caplog):
    mock_prepare_datasets.side_effect = Exception("Dataset not found")

    request_data = {
        "model_name": "RFC",
        "dataset": "NonExistentDataset",
        "model_params": {"n_estimators": 10}
    }

    with caplog.at_level(logging.ERROR):
        response = client.post("/train", json=request_data)

    assert response.status_code == 200
    assert "error" in response.json()
    assert response.json()["error"] == "Dataset not found"

    assert any("Dataset NonExistentDataset_train not found" in record.message for record in caplog.records), "Dataset not found log not found"

def test_get_status():
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == {"status": "Train API is up"}
