import pytest
from fastapi.testclient import TestClient
from update import app
import logging
import re

#This script works with regex expressions to check for similiar log messages or to include placeholders for versions / model_names. 
# This functionality can be included in the other scripts if needed.

client = TestClient(app)

def test_get_status():
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == {"status": "Update API is up"}

@pytest.mark.parametrize(
    "model_name, dataset, metric_name, log_level, expected_status, expected_error, expected_log_messages",
    [
        (
            "RFC",
            "Ptbdb",
            "accuracy",
            logging.INFO,
            200,
            None,
            [
                "Versions found in set_deployment_alias",
                "Set 'not_deployment' alias for version .*",
                "Set version .* as deployment for model .*"
            ]
        ),
        (
            "RFC",
            "Ptbdb",
            "non_existing_metric",
            logging.ERROR,
            200,
            "metric_name is not defined / available",
            [
                "Error during updating: metric_name is not defined / available"
            ]
        ),
        (
            "NonExistentModel",
            "Ptbdb",
            "accuracy",
            logging.ERROR,
            200,
            "model / dataset combination could not be found in the MLFlow model registry",
            [
                "Error during updating: model / dataset combination could not be found in the MLFlow model registry"
            ]
        ),
        (
            "RFC",
            "NonExistentDataset",
            "accuracy",
            logging.ERROR,
            200,
            "model / dataset combination could not be found in the MLFlow model registry",
            [
                "Error during updating: model / dataset combination could not be found in the MLFlow model registry"
            ]
        ),
    ]
)

def test_make_update(caplog, model_name, dataset, metric_name, log_level, expected_status, expected_error, expected_log_messages):
    with caplog.at_level(log_level):
        response = client.post("/update", json={"model_name": model_name, "dataset": dataset, "metric_name": metric_name})

    assert response.status_code == expected_status

    if expected_error:
        assert "Error during updating" in response.json()
        assert expected_error in response.json()["Error during updating"]
    else:
        assert "deployment_model set to" in response.json()

    print(f"logs from function with model_name {model_name}, dataset {dataset}, and metric_name {metric_name}:")
    print([record.message for record in caplog.records])

    for expected_log_message in expected_log_messages:
        pattern = re.compile(expected_log_message)
        assert any(pattern.search(record.message) for record in caplog.records), f"Pattern '{expected_log_message}' not found in logs"
