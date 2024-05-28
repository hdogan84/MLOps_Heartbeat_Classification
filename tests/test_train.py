#this is the test-script for the train endpoint

### EXAMPLE CODE:
import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from src.app.train import app, TrainModelRequest

client = TestClient(app)

class TestTrainAPI(unittest.TestCase):

    @patch("src.app.train.prepare_datasets")
    @patch("src.app.train.download_datasets")
    def test_train_model_success(self, mock_download_datasets, mock_prepare_datasets):
        # Mocking the dataset preparation
        mock_prepare_datasets.return_value = {
            "X_train_Mitbih": [[0.1, 0.2, 0.3]] * 10,
            "y_train_Mitbih": [0, 1] * 5,
            "X_test_Mitbih": [[0.1, 0.2, 0.3]] * 4,
            "y_test_Mitbih": [0, 1] * 2
        }
        
        request_data = {
            "model_name": "RFC",
            "dataset": "Mitbih",
            "model_params": {"n_estimators": 10}
        }
        
        response = client.post("/train", json=request_data)
        self.assertEqual(response.status_code, 200)
        self.assertIn("status", response.json())
        self.assertIn("trained", response.json()["status"])

    @patch("src.app.train.prepare_datasets")
    @patch("src.app.train.download_datasets")
    def test_train_model_dataset_not_found(self, mock_download_datasets, mock_prepare_datasets):
        # Mocking an empty dataset preparation
        mock_prepare_datasets.side_effect = Exception("Dataset not found")
        
        request_data = {
            "model_name": "RFC",
            "dataset": "NonExistentDataset",
            "model_params": {"n_estimators": 10}
        }
        
        response = client.post("/train", json=request_data)
        self.assertEqual(response.status_code, 200)
        self.assertIn("error", response.json())
        self.assertEqual(response.json()["error"], "Dataload failed")

    def test_get_status(self):
        response = client.get("/status")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "Train API is up"})

if __name__ == "__main__":
    unittest.main()

