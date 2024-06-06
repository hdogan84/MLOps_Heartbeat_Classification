#what should be tested for the gateway-api
    # status of gateway api itself
    # fastapi-users functionalities
        # create / register user (with adm rights)
        # sign in / log off
        # delete user
        # check user database / print it out
    # Connection to MLFlow server (Mlflow-client)
        # show model-registry content (models) --> Print it out. (pprint)
        # Delete entire model-registry aka all models (and show / test that its deleted) --> Preparation for the end-to-end-tets / integration tests // OR: Get the latest version for later assertion of training models.
    # Workflow testing (tested on the endpoints / sub-apis!)
        #### HACK: If mlfow testing is not implemented, just delete the content (but not the folder itself!) of mlruns folder on main level.
        # train model RFC on PTBDB (Must be version 1 if the model-registry is deleted --> assert Version 1 hardcoded!) with bad model_params --> Low accuracy! assert it with the model_registry
        # train model RFC on Ptbdb (Version 2) with good model_params --> High accuracy
        # update the RFC_Ptbdb Model to deployment --> Version 2 must be set as deployment --> Assert it
        # predict with RFC deployment model on Ptbdb --> Must give the correct status etc, assert it just like in test_predict.py, but not as exhausting (just the correct prediction)



#### Some useful links for instructions:
    # https://stackoverflow.com/questions/75466872/integration-testing-fastapi-with-user-authentication --> user authentification testing
    # https://fastapi-users.github.io/fastapi-users/12.1/usage/flow/ --> Setting superuser is only possible on database level!

### ONGOING NOTES ###
# Superusers cannot be created via /auth/register or on startup --> Patch-function / Endpoint does this? Try it out.
import pytest
from fastapi.testclient import TestClient
from gateway_app import app as application
from user_db import User, create_db_and_tables, get_user_db

@pytest.fixture(scope="module")
def client():
    with TestClient(application) as c:
        yield c

class TestTokens:
    """
    Generating the tokens once with signing in and then storing them --> Behavior of cookies / authorization headers is mimicked.
    Avoids the error, that for each function a different token is generated trough the mandatory signing in.
    """
    test_user_token = None
    admin_user_token = None

@pytest.fixture
def test_user():
    return {
        "email": "testuser@example.com",
        "password": "password123",
        "is_active": "true",
        "is_superuser": "true",
        "is_verified": "true"
    }

@pytest.fixture
def admin_user():
    return {
        "email": "admin@example.com",
        "password": "admin",
        "is_active": "true",
        "is_superuser": "true",
        "is_verified": "true"
    }

def test_get_status(client):
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == {"status": 1}

def test_create_user_test_user(client, test_user):
    response = client.post("/auth/register", json=test_user)
    try:
        assert response.status_code == 201
        print("New user successfully created. Checking other response contents.")
        assert response.json()["email"] == test_user["email"]
        try:
            assert response.json()["is_active"] == True
        except Exception as e:
            print("activation the email via post is not possible? Trying other response (False)")
            assert response.json()["is_active"] == False
        try:
            assert response.json()["is_superuser"] == test_user["is_superuser"]
        except Exception as e:
            print("Creation of superuser is not possible with register route. Trying other response (False).")
            assert response.json()["is_superuser"] == False
        try:
            assert response.json()["is_verified"] == test_user["is_verified"]
        except Exception as e:
            print("verification the email via post is not possible? Trying other response (False)")
            assert response.json()["is_verified"] == False
    except Exception as e:
        assert response.status_code == 400
        assert response.json()["detail"] == "REGISTER_USER_ALREADY_EXISTS"
        print("User already exists. No further content is provided.")

def test_sign_in_test_user(client, test_user):
    login_data = {
        "username": test_user["email"],
        "password": test_user["password"],
    }
    headers = {"accept": "application/json", "Content-Type": 'application/x-www-form-urlencoded'}
    response = client.post("/auth/jwt/login", data=login_data, headers=headers)
    assert response.status_code == 200
    assert "access_token" in response.json()
    TestTokens.test_user_token = response.json()["access_token"]

def test_authenticated_route_as_test_user(client, test_user):
    headers = {"accept": 'application/json', "Authorization": f"Bearer {TestTokens.test_user_token}"}
    response = client.get("/authenticated-route", headers=headers)
    assert response.status_code == 200
    assert response.json() == {"message": f"Hello {test_user['email']}, you are not a superuser"}

def test_log_out_test_user(client):
    headers = {"accept": 'application/json', "Authorization": f"Bearer {TestTokens.test_user_token}"}
    response = client.post("/auth/jwt/logout", headers=headers)
    try:
        assert response.status_code == 200
    except Exception as e:
        print("response code for logging out is not 200, trying to assert 204.")
        assert response.status_code == 204
        print("status is undocumented for logging out.")

def test_sign_in_admin(client, admin_user):
    login_data = {
        "username": admin_user["email"],
        "password": admin_user["password"],
    }
    headers = {"accept": "application/json", "Content-Type": 'application/x-www-form-urlencoded'}
    response = client.post("/auth/jwt/login", data=login_data, headers=headers)
    assert response.status_code == 200
    assert "access_token" in response.json()
    TestTokens.admin_user_token = response.json()["access_token"]

def test_authenticated_route_as_admin(client, admin_user):
    headers = {"accept": 'application/json', "Authorization": f"Bearer {TestTokens.admin_user_token}"}
    response = client.get("/authenticated-route", headers=headers)
    assert response.status_code == 200
    assert response.json() == {"message": f"Hello {admin_user['email']}, you are a superuser"}

def test_log_out_admin(client):
    headers = {"accept": 'application/json', "Authorization": f"Bearer {TestTokens.admin_user_token}"}
    response = client.post("/auth/jwt/logout", headers=headers)
    try:
        assert response.status_code == 200
    except Exception as e:
        print("response code for logging out is not 200, trying to assert 204.")
        assert response.status_code == 204
        print("status is undocumented for logging out.")


"""
def test_delete_user(test_user): #Only possible as superuser / admin --> Needs the id that can (only?) be retrieved from users/me endpoint and thus while logged in as test_user?
    # First, sign in to get the access token
    login_data = {
        "username": test_user["email"],
        "password": test_user["password"],
    }
    response = client.post("/auth/jwt/login", data=login_data)
    access_token = response.json()["access_token"]

    # Delete the user
    headers = {"Authorization": f"Bearer {access_token}"}
    response = client.delete("/users/me", headers=headers)
    assert response.status_code == 204

def test_check_user_database():
    response = client.get("/users")
    assert response.status_code == 200
    users = response.json()
    assert isinstance(users, list)"""






#####+ AMIR START HERE WITH WORKFLOW TESTING --> Recycle the sign-in process etc. We might protect some Endpoints with current_active_user == is_superuser for further checks and security.###########

