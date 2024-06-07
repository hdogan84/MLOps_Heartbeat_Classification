import pytest
from fastapi.testclient import TestClient
from predict import app
import logging

client = TestClient(app)

@pytest.mark.parametrize(
    "model_name, dataset, log_level, expected_status, expected_response_key, expected_response_value, expected_log_messages",
    [
        (
            "RFC",
            "Mitbih",
            logging.INFO,
            200,
            "prediction",
            None,
            ["Predictions made successfully.", "predictions with ML-Model from deployment made successfully"]
        ),
        (
            "NonExistentModel",
            "Mitbih",
            logging.ERROR,
            200,
            "Error during prediction",
            "RESOURCE_DOES_NOT_EXIST: Registered Model with name=NonExistentModel_Mitbih not found",
            ["Error during prediction: RESOURCE_DOES_NOT_EXIST: Registered Model with name=NonExistentModel_Mitbih not found"]
        ),
        (
            "RFC",
            "NonExistentDataset",
            logging.ERROR,
            200,
            "Error during prediction",
            "RESOURCE_DOES_NOT_EXIST: Registered Model with name=RFC_NonExistentDataset not found",
            ["Error during prediction: RESOURCE_DOES_NOT_EXIST: Registered Model with name=RFC_NonExistentDataset not found"]
        ),
    ]
)
def test_make_prediction(caplog, model_name, dataset, log_level, expected_status, expected_response_key, expected_response_value, expected_log_messages):
    with caplog.at_level(log_level):
        response = client.post("/predict", json={"model_name": model_name, "dataset": dataset})
    
    assert response.status_code == expected_status
    
    if expected_response_key == "prediction":
        assert "prediction" in response.json()
        assert "true_value" in response.json()
    else:
        assert expected_response_key in response.json()
        assert expected_response_value in response.json()[expected_response_key]

    print(f"logs from function with model_name {model_name} and dataset {dataset}:")
    print([record.message for record in caplog.records])

    for expected_log_message in expected_log_messages:
        assert any(expected_log_message in record.message for record in caplog.records), f"{expected_log_message} not found in logs"

def test_get_status():
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == {"status": "Prediction API is up"}
