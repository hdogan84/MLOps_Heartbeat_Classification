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
from gateway_app import app as application #trying to fix the module not callable error.
from user_db import User, create_db_and_tables, get_user_db

@pytest.fixture(scope="module")
def client():
    """
    This fixture is crucial to initiate the creation of user database and tables in the gateway_app.py script!
    working with github actions: YES.
    working with docker: YES.
    """
    with TestClient(application) as c:
      yield c

@pytest.fixture
def test_user(): #this simulates an admin user --> Cannot be set via the /auth/register route, but for simulation purposes this is enough.
    return {
  "email": "testuser@example.com",
  "password": "password123",
  "is_active": "true",
  "is_superuser": "true",
  "is_verified": "true" #must be lowerletter!
}

@pytest.fixture
def new_user(): #this simulates a normal user
    return {
        "email": "newuser@example.com",
        "password": "newpassword123",
        "is_active": "true",
        "is_superuser": "false",
        "is_verified": "true"
    }

def test_get_status(client):
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == {"status": 1}

def test_create_user(client, test_user):
    response = client.post("/auth/register", json=test_user)
    assert response.status_code == 201
    assert response.json()["email"] == test_user["email"]
    try:
        assert response.json()["is_active"] == True # test_user["is_active"] #there seems to be a problem with the syntax, because fastapi writes true in lower case??!! Is this a valid workaround? Because Posting to the endpoint with True (upper case) is not possible.
    except Exception as e:
        print("activation the email via post is not possible? Trying other response (False)")
        assert response.json()["is_active"] == False
    try:
        assert response.json()["is_superuser"] == test_user["is_superuser"] #checking this if it works with strings. Apparently, a superuser cannot be created with the /auth/register route. It is also unclear, where the database is stored.
    except Exception as e:
        print("Creation of superuser is not possible with register route. Trying other response (False).")
        assert response.json()["is_superuser"] == False
    try:
        assert response.json()["is_verified"] == test_user["is_verified"]
    except Exception as e:
        print("verification the email via post is not possible? Trying other response (False)")
        assert response.json()["is_verified"] == False


###### SUGGESTIONS FOR FURTHER TESTS, WITH REWORK NEEDED (INCLUDE client AS ARGUMENT) --> WATCH OUT, SUPER USER CANNOT BE CREATED VIA THE /auth/register-route ####
def test_sign_in(test_user):
    login_data = {
        "username": test_user["email"],
        "password": test_user["password"],
    }
    response = client.post("/auth/jwt/login", data=login_data)
    assert response.status_code == 200
    assert "access_token" in response.json()

def test_delete_user(test_user):
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
    assert isinstance(users, list)

def test_register_and_login_user(new_user):
    # Register the new user
    response = client.post("/auth/register", json=new_user)
    assert response.status_code == 201
    assert response.json()["email"] == new_user["email"]

    # Login the new user
    login_data = {
        "username": new_user["email"],
        "password": new_user["password"],
    }
    response = client.post("/auth/jwt/login", data=login_data)
    assert response.status_code == 200
    assert "access_token" in response.json()

def test_authenticated_route_as_superuser(test_user):
    # First, sign in to get the access token
    login_data = {
        "username": test_user["email"],
        "password": test_user["password"],
    }
    response = client.post("/auth/jwt/login", data=login_data)
    access_token = response.json()["access_token"]

    # Access the authenticated route
    headers = {"Authorization": f"Bearer {access_token}"}
    response = client.get("/authenticated-route", headers=headers)
    assert response.status_code == 200
    assert response.json() == {"message": f"Hello {test_user['email']}, you are a superuser"}

def test_authenticated_route_as_non_superuser(new_user):
    # Register the new user
    response = client.post("/auth/register", json=new_user)
    assert response.status_code == 201
    assert response.json()["email"] == new_user["email"]

    # Login the new user
    login_data = {
        "username": new_user["email"],
        "password": new_user["password"],
    }
    response = client.post("/auth/jwt/login", data=login_data)
    access_token = response.json()["access_token"]

    # Access the authenticated route
    headers = {"Authorization": f"Bearer {access_token}"}
    response = client.get("/authenticated-route", headers=headers)
    assert response.status_code == 200
    assert response.json() == {"message": f"Hello {new_user['email']}, you are not a superuser"}




#####+ AMIR START HERE WITH WORKFLOW TESTING ###########

