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

@pytest.mark.parametrize(
    "request_data, mock_side_effect, log_level, expected_status, expected_response_key, expected_response_value, expected_log_message",
    [
        (
            {
                "model_name": "RFC",
                "dataset": "Ptbdb",
                "model_params": {"n_estimators": 10}
            },
            None,
            logging.INFO,
            200,
            "status",
            "trained",
            "------------------------------Model training successful---------------------------------"
        ),
        (
            {
                "model_name": "RFC",
                "dataset": "NonExistentDataset",
                "model_params": {"n_estimators": 10}
            },
            Exception("Dataset not found"),
            logging.ERROR,
            200,
            "error",
            "Dataset not found",
            "Dataset NonExistentDataset_train not found"
        ),
    ]
)
def test_train_model(mock_prepare_datasets, caplog, request_data, mock_side_effect, log_level, expected_status, expected_response_key, expected_response_value, expected_log_message):
    if mock_side_effect:
        mock_prepare_datasets.side_effect = mock_side_effect

    with caplog.at_level(log_level):
        response = client.post("/train", json=request_data)

    assert response.status_code == expected_status
    assert expected_response_key in response.json()
    assert response.json()[expected_response_key] == expected_response_value

    print(f"logs from function with request_data {request_data}:")
    print([record.message for record in caplog.records])

    assert any(expected_log_message in record.message for record in caplog.records), f"{expected_log_message} not found in logs"

def test_get_status():
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == {"status": "Train API is up"}
