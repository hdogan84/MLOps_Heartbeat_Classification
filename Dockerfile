# Dockerfile for testing --> Does the same tests as github actions but inside the containers, called by docker-compose.test.yml!!! Not docker-compose.yml, which builds the normal image / containers.
FROM python:3.11

WORKDIR /app

COPY . /app

RUN pip install -r src/app/requirements.txt
RUN pip install pytest

CMD ["pytest", "tests"]