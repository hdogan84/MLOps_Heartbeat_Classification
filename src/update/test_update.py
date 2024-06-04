#this script will update the RFC_Ptbdb model, because this is also the model, that is trained in the training script (for consistency)
import pytest
from fastapi.testclient import TestClient
from update import app  # Assuming the provided script is saved in a file named `app.py`

client = TestClient(app)

def test_get_status():
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == {"status": "Update API is up"}

@pytest.mark.parametrize("model_name, dataset, metric_name, expected_status, expected_error", [
    ("RFC", "Ptbdb", "accuracy", 200, None),
    ("RFC", "Ptbdb", "non_existing_metric", 200, "metric_name is not defined / available"),
    ("NonExistentModel","Ptbdb", "accuracy", 200, "model / dataset combination could not be found in the MLFlow model registry"),
    ("RFC", "NonExistentDataset", "accuracy", 200, "model / dataset combination could not be found in the MLFlow model registry")
])

def test_make_update(model_name, dataset, metric_name, expected_status, expected_error):
    response = client.post("/update", json={"model_name": model_name, "dataset": dataset, "metric_name": metric_name})
    assert response.status_code == expected_status
    
    if expected_error:
        assert "Error during updating" in response.json()
        assert expected_error in response.json()["Error during updating"]
    else:
        assert "prediction" in response.json()
        assert "true_value" in response.json()

#special notes for update.py / test_update.py etc:
    # naming convention is now the same as in all other scripts.
    # It is not entirely clear if the model is not available or the dataset --> Use the advanced logging evaluation for this.


# more logging evaluations just like in test_train.py should be done, because the error from MLFlow is always related to the model,
# and its not entirely clear, if the dataset is included in the datasets-dictionary!
# especially for checking if the dataset is available, the logs must be evaluated, because the code tries to load the model (with _NonExistentDataset)
# before the check for the existance of the dataset is performed.

# Reuse the mark.parametrize logic to include the logging message values?