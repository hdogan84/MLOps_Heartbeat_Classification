import pytest
from fastapi.testclient import TestClient
#from src.train.train import app #this is only valid if the train.py is only available in sr/train/train.py 
from train import app #this is valid for the github actions, if the test_train.py and train.py are in the same folder (container)
##--> Produces import error on github, but in the example, it is the same syntax!!: https://github.com/DataScientest-Studio/juin23_continu_mlops_pompiers/tree/master/src/api_user
from unittest.mock import patch, MagicMock
import warnings
import logging

warnings.filterwarnings("ignore", category=DeprecationWarning) #this makes the depreceation warnings disappear, because they cannot be solved easily and require an migration of different packages. HOWEVER, they do not disturb the functioning of the code!

client = TestClient(app)


#Explanation for commenting out so much:
#The mocking functions would be useful, if the unit tests would be performed without the other containers running. But, also refering to:https://github.com/DataScientest-Studio/juin23_continu_mlops_pompiers/blob/master/.github/workflows/python-app.yml
# The unit tests can also be performed inside a container, which makes even more sence because it saves a lot of work regarding the recreation of mocking functions and tests the whole API structure and connection to MLFlow in one go
# The end_to_end / integration test can then be performend as a "suggested workflow" in the gateway api (test_app.py). 

# Utility to capture log output --> Not sure how and if this works, but could be useful for integration testing.
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
    logger = logging.getLogger("train")
    logger.addHandler(log_capture)
    yield log_capture
    logger.removeHandler(log_capture)


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

def test_train_model_success(): #this uses the smaller Ptbdb Dataset and NO mock dataset, but its fast enough to check.
    request_data = {
        "model_name": "RFC",
        "dataset": "Ptbdb",
        "model_params": {"n_estimators": 10} #We even can pass model_params as a dict :)
    }

    response = client.post("/train", json=request_data)
    assert response.status_code == 200
    # assert  "Dataload successfull" in logging.info 
    assert "status" in response.json()
    assert response.json()["status"] == "trained"
    
    #this log capture could be running falsely, but if it works this is a great workaround for unit testing (collecting the necessary log-messages after each successful function)
    logs = log_capture.get_logs()
    print("logs from function test_train_model_success:")
    print(logs)
    assert any("------------------------------Model training successful---------------------------------" in log for log in logs), "Model training log not found"

def test_train_model_dataset_not_found(mock_prepare_datasets):
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



##### NOTES TO MAKE IT TO UNIT -TESTING ####
# --> Evaluate the logging.info messages!
# if this is too complicated:
# --> return message has some info about the succession of each function used in the script (dict).