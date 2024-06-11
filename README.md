Heartbeat Classification Microservice API
==============================

This repository contains a comprehensive system for monitoring, training, and predicting heartbeats using various machine learning models. The system is built using FastAPI and includes multiple (micro)services for training, updating, and predicting models, all orchestrated using Docker Compose.  All tasks from the sub-apis are executed as background tasks so the Gateway-API stays open for more requests, although limited by RateLimiter.

This Readme focuses on the global context of the project and the main functions of the Gateway-API, which orchestrates the sub-apis. The sub-apis are explained more detailled in the corresponding folder and the readme.md-files in those folders.


## Table of Contents
- [Project Organization](#project-organization) 
- [Flowchart](#flowchart)
- [Features (Overview)](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Application](#running-the-application)
- [Usage](#usage-using-the-gateway-api-docs-interface)
  - [Authentication](#authentication)
  - [Training](#training-only-for-admin-users)
  - [Updating Models](#updating-models-only-for-admin-users)
  - [Predicting](#predicting-for-all-users)
- [Running Tests](#running-tests-live-on-docker)
- [CI/CD Pipeline](#cicd-pipeline)
  - [Triggering the CI/CD Pipeline](#triggering-the-cicd-pipeline)
  - [CI/CD Workflow Configuration](#cicd-workflow-configuration)


## Project Organization 

------------
.
├── Dockerfile                                 <- Docker configuration for building the project
├── LICENSE                                    <- License file for the project
├── README.md                                  <- Project readme file
├── docker-compose.yaml                        <- Docker Compose configuration file
├── mlruns                                     <- Directory for MLflow experiment logs
├── notebooks                                  <- Jupyter notebooks for experiments and analysis
├── project_structure.txt                      <- File describing the project structure
├── project_structure_2.txt                    <- Another file describing the project structure
├── references                                 <- Directory for reference materials
├── reports                                    <- Directory for reports
│   ├── MLOps_Heartbeat_Canvas.docx            <- The canvas used to write the specifications
│   ├── REPORT_Specifications_MLOps_Projects.pdf <- The specifications for this project
│   ├── figures                                <- Directory for storing figures
│   └── logs                                   <- Directory for log files
│       └── app.log                            <- Application log file
└── src                                        <- Source code directory
    ├── __init__.py                            <- Initialization file for the src package
    ├── app                                    <- Application code directory
    │   ├── Dockerfile                         <- Docker configuration for the app service
    │   ├── __init__.py                        <- Initialization file for the app package
    │   ├── gateway_app.py                     <- Gateway application script
    │   ├── kaggle.json                        <- Kaggle API credentials
    │   ├── requirements.txt                   <- Python dependencies for the app service
    │   ├── test_app.py                        <- Tests for the app service
    │   ├── user_db.py                         <- Database management for user data
    │   ├── user_schemas.py                    <- Schema definitions for user data
    │   └── users.py                           <- User management script
    ├── config                                 <- Configuration files directory
    ├── data                                   <- Data processing and storage directory
    │   ├── __init__.py                        <- Initialization file for the data package
    │   └── heartbeat                          <- Heartbeat data directory
    │       ├── mitbih_test.csv                <- MIT-BIH test dataset
    │       ├── mitbih_train.csv               <- MIT-BIH train dataset
    │       ├── ptbdb_abnormal.csv             <- PTBDB abnormal dataset
    │       └── ptbdb_normal.csv               <- PTBDB normal dataset
    ├── mlflow                                 <- MLflow server directory
    │   ├── Dockerfile                         <- Docker configuration for MLflow server
    │   ├── __init__.py                        <- Initialization file for the mlflow package
    │   └── mlflow-server.sh                   <- Script to start the MLflow server
    ├── predict                                <- Prediction service directory
    │   ├── Dockerfile                         <- Docker configuration for the predict service
    │   ├── __init__.py                        <- Initialization file for the predict package
    │   ├── predict.py                         <- Prediction script
    │   ├── requirements.txt                   <- Python dependencies for the predict service
    │   └── test_predict.py                    <- Tests for the predict service
    ├── predict_simulation_NOT_FINISHED        <- Prediction simulation service (incomplete)
    │   ├── Dockerfile                         <- Docker configuration for the simulation service
    │   ├── __init__.py                        <- Initialization file for the simulation package
    │   ├── predict_simulation.py              <- Prediction simulation script
    │   ├── requirements.txt                   <- Python dependencies for the simulation service
    │   └── test_predict_simulation.py         <- Tests for the simulation service
    ├── train                                  <- Training service directory
    │   ├── Dockerfile                         <- Docker configuration for the train service
    │   ├── __init__.py                        <- Initialization file for the train package
    │   ├── kaggle.json                        <- Kaggle API credentials for training
    │   ├── requirements.txt                   <- Python dependencies for the train service
    │   ├── test_train.py                      <- Tests for the train service
    │   └── train.py                           <- Training script
    └── update                                 <- Update service directory
        ├── Dockerfile                         <- Docker configuration for the update service
        ├── __init__.py                        <- Initialization file for the update package
        ├── requirements.txt                   <- Python dependencies for the update service
        ├── test_update.py                     <- Tests for the update service
        └── update.py                          <- Update script

--------
## Flowchart
![Alt text](https://github.com/hdogan84/May24_MLOps_Heartbeat_Classification/blob/development/reports/figures/Flowchart_Gateway_api_11_06_24.png?raw=true "Architecture of the gateway-api with fastapi_users authentication and connections to / functionalities of the sub-apis.")


## Features
- **Authentication**: User management with JWT authentication.
- **Prediction**: Real-time prediction requests handled asynchronously.
- **Training**: Background training of models with configurable parameters.
- **Model Update**: Update deployed models based on specified metrics.
- **Rate Limiting**: Protect endpoints with rate limiting to prevent abuse.
- **Monitoring**: All relevant function outputs or steps are logged in one comprehensive 'app.log' logfile.

More information on the specific functionalities of the sub-apis
- Prediction
- Training
- Model Update

can be found in the readme-files for these sub-apis.


## Getting Started

### Prerequisites
- Docker
- Docker Compose

### Installation
1. Clone the repository and go into the directory:
   ```sh
   git clone https://github.com/hdogan84/May24_MLOps_Heartbeat_Classification.git
   cd May24_MLOps_Heartbeat_Classification
### Running the application
1. Build and start the services using Docker Compose
    ```sh
    docker compose up --build -d
2. The following services will be available (only gateway API docs interface is necessary for interaction)
- Gateway API: http://localhost:8000/docs
- Training API: http://localhost:8001
- Update API: http://localhost:8002
- Predict API: http://localhost:8003
- MLflow server: http://localhost:5000

## Usage (using the gateway-api docs interface)
### Authentication
By default, an admin user is created by hard-writing in the users database on startup of the gateway app with username / email 'admin@example.com' and password 'admin'. This is due to the restriction on the following endpoints, which do not allow new admin users to be created.
* register a new user
    ```sh
    POST /auth/register
* Login
    
    use either the 'Authorization' Button on the top right of the /docs interface of the gateway-api (recommended) or use the coding route with
    ```sh
    Post /auth/jwt/login
* Access protected routes

    if authorized using the 'Authorize' Button: Just use the different endpoints without submitting Authorization Headers manually.

    If authorized using the 
    ```sh
    Post /auth/jwt/login
    ```
    method, submit the gathered Authorization headers in your POST requests as follows (example):
    ```sh
    headers = {
        "accept": "application/json",
        "Authorization": "Bearer YOUR_RECEIVED_AUTHORIZATION_TOKEN"
    }
    ```

### Training (Only for admin users)
This function calls the sub-api for training the models.

```sh
Post /train
```
At this timepoint, only RFC-models are trainable. In the request body, all possible hyperparameters can be passed as **kwargs to the training function. Standard Hyperparameter is
```sh
{"n_jobs": -1}
```

### Updating Models (Only for admin users)
The updating function will call the sub-api for generating the deployment alias / updating the best model according to a specified metric (standard is accuracy).
```sh
Post /update
```
The models in the MLflow model-registry will be evaluated and will be given an according alias.

### Predicting (for all users)
With the prediction function, the sub-api for prediction is called.
```sh
Post /predict
```
Since this project relies on static data, only a simulation of the evaluation of a single heartbeat is implemented: A random row from the selected dataset is given to the prediction function. This prediction function will use the model from the MLflow model-registry that is currently set as the deployment model.


## Running tests (live on docker)
Pytests are written for all APIs, including the sub-apis. They can be run manually with docker using the following routine.
1. Ensure the services are running in docker
    ```sh
    docker compose up --build -d #only if not done before
    docker ps --all #show all running containers
    ```
    This should give:
    ```
    CONTAINER ID   IMAGE                COMMAND                  CREATED      STATUS      PORTS                    NAMES
    0a6f9ab28251   gateway-api:latest   "uvicorn gateway_app…"   3 days ago   Up 3 days   0.0.0.0:8000->8000/tcp   may24_mlops_heartbeat_classification-gateway-api-1
    c9e67f0fd3d7   train-api:latest     "uvicorn train:app -…"   3 days ago   Up 3 days   0.0.0.0:8001->8001/tcp   may24_mlops_heartbeat_classification-train-api-1
    351af6ac1073   predict-api:latest   "uvicorn predict:app…"   3 days ago   Up 3 days   0.0.0.0:8003->8003/tcp   may24_mlops_heartbeat_classification-predict-api-1
    6104bde93b55   update-api:latest    "uvicorn update:app …"   3 days ago   Up 3 days   0.0.0.0:8002->8002/tcp   may24_mlops_heartbeat_classification-update-api-1
    b17b2043b453   redis:alpine         "docker-entrypoint.s…"   3 days ago   Up 3 days   0.0.0.0:6379->6379/tcp   redis
    75bc55ca072a   mlflow_server        "mlflow server --bac…"   3 days ago   Up 3 days   0.0.0.0:5000->5000/tcp   mlflow_server
    ```
2. Select the container and corresponding test that you wish to run (example for gateway-api with the corresponding test 'test_app.py')
    ```sh
    docker compose run --workdir /app gateway-api python -m pytest -s -v test_app.py
    ```
    In this example, you can watch the routine of the test on the mlflow-webserver (two new models are trained, updated and a prediction is made). The app.log logfile also gives an comprehensive overview on the routines that work underlying.

## CI/CD Pipeline
The Ci/CD pipeline is set up using Github Actions. It includes steps for linting, testing and building / pushing the Docker images to dockerhub.
* **Linting**: Ensures code quality using flake8. Stops the build if syntax errors occur.
* **Testing**: Runs pytest on all APIs
* **Building**: If all tests with pytest are succesfull, the Docker images for all APIs are built and deployed on dockerhub.

### Triggering the CI/CD Pipeline
To trigger the pipeline, push changes to the 'master' or 'development' branch or create a pull request on those branches.

### CI/CD Workflow Configuration
The CI/CD workflow is defined in '.github/workflows/ci-cd.yml'.



<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
