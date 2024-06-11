#DOCKERFILE training service with train.py (app = FastAPI() inside)
FROM python:3.11.0rc1

WORKDIR /app

COPY requirements.txt requirements.txt
#COPY kaggle.json: WARNING: THIS IS SIMONS PERSONAL KEY, REMOVE BEFORE DEPLOYMENT / PUBLISHING.
COPY kaggle.json /root/.kaggle/kaggle.json

RUN pip install -r requirements.txt

#install pytest in the container --> Necessary for github actions?
RUN pip install pytest

COPY . .

CMD ["uvicorn", "train:app", "--host", "0.0.0.0", "--port", "8001"]