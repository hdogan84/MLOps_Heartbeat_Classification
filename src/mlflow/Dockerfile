#DOCKERFILE mlflow-server service with bash file
FROM python:3.11.0rc1

WORKDIR /app

RUN pip install mlflow boto3

COPY mlflow-server.sh .

# give rights to execute
RUN chmod +x mlflow-server.sh

CMD ["./mlflow-server.sh"]