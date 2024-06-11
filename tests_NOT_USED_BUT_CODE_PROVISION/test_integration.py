#this is the integration testing (tests that different modules work together)
##EXAMPLE CODE:
import pytest
from . import gateway_app
from src.data import load_data

def test_app_data_integration():
    data = load_data()
    result = gateway_app.process_data(data)
    assert result is not None