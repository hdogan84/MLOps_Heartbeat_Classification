#DOCKERFILE prediction service with predict.py (app = FastAPI() inside)
FROM python:3.11.0rc1

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8003"]