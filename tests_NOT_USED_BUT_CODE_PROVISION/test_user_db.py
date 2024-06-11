#this is the test-script for testing the user_db from fastapi users

## EXAMPLE CODE:
# tests/test_user_db.py
import pytest
from src.app.user_db import get_user_by_id

def test_get_user_by_id():
    user_id = 1
    user = get_user_by_id(user_id)
    assert user.id == user_id