import pytest
from fastapi.testclient import TestClient
from predict import app  # Assuming the provided script is saved in a file named `app.py`

client = TestClient(app)

def test_get_status():
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == {"status": "Prediction API is up"}

@pytest.mark.parametrize("model_name, dataset, expected_status, expected_error", [
    ("RFC", "Mitbih", 200, None),
    ("NonExistentModel", "Mitbih", 200, "RESOURCE_DOES_NOT_EXIST: Registered Model with name=NonExistentModel_Mitbih not found"),
    ("RFC", "NonExistentDataset", 200, "RESOURCE_DOES_NOT_EXIST: Registered Model with name=RFC_NonExistentDataset not found")
])
def test_make_prediction(model_name, dataset, expected_status, expected_error):
    response = client.post("/predict", json={"model_name": model_name, "dataset": dataset})
    assert response.status_code == expected_status
    
    if expected_error:
        assert "Error during prediction" in response.json()
        assert expected_error in response.json()["Error during prediction"]
    else:
        assert "prediction" in response.json()
        assert "true_value" in response.json()

# more logging evaluations just like in test_train.py should be done, because the error from MLFlow is always related to the model,
# and its not entirely clear, if the dataset is included in the datasets-dictionary!
# especially for checking if the dataset is available, the logs must be evaluated, because the code tries to load the model (with _NonExistentDataset)
# before the check for the existance of the dataset is performed.
