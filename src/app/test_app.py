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


import pytest
from fastapi.testclient import TestClient
from gateway_app import app as application #trying to fix the module not callable error.
from user_db import User, create_db_and_tables, get_user_db
#from fastapi_users.manager import UserAlreadyExists #this is not used and also not available as module and produces errors.

client = TestClient(application) #trying to avoid module calling error?

@pytest.fixture(scope="module", autouse=True) #we have to create the databases just like it would happen on startup of gateway_app.py (lifespan handler)
async def setup_database():
    await create_db_and_tables()
    yield
    #optionally: Clean the database after testing.

@pytest.fixture
def test_user(): #this simulates an admin user
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

def test_get_status():
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == {"status": 1}

def test_create_user(test_user):
    response = client.post("/auth/register", json=test_user)
    assert response.status_code == 201
    assert response.json()["email"] == test_user["email"]
    assert response.json()["is_active"] == test_user["is_active"]
    #the assertion of superuser is not necessary now, because apparently it is not possible to create a superuser via the register route? So this check leads only to problems.
    #assert response.json()["is_superuser"] == "false" #checking this if it works with strings. Apparently, a superuser cannot be created with the /auth/register route. It is also unclear, where the database is stored.
    assert response.json()["is_verified"] == test_user["is_verified"]






#####+ AMIR START HERE WITH WORKFLOW TESTING ###########

