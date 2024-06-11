#DOCKERFILE update service with update.py (app = FastAPI() inside) (essentially deployment)
FROM python:3.11.0rc1

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8002

CMD ["uvicorn", "update:app", "--host", "0.0.0.0", "--port", "8002"]