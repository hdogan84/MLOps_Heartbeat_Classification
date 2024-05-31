#this is the integration testing (tests that different modules work together)
##EXAMPLE CODE:
import pytest
from src.app import app
from src.data import load_data

def test_app_data_integration():
    data = load_data()
    result = app.process_data(data)
    assert result is not None