#this is the end_to_end test script (checks the whole workflow)
## MUST BE TESTED WITHOUT MLFLOW SERVER, SO MODELS HAVE TO BE CREATED MANUALLY IN THIS FILE?!

## Example code:

import requests

def test_end_to_end():
    response = requests.post('http://localhost:5000/api/v1/predict', json={'data': 'test_data'})
    assert response.status_code == 200
    assert 'prediction' in response.json()

