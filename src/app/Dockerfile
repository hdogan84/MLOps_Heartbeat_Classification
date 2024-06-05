#this dockerfile is written for the v1 of the docker-apis: Only one gateway api. this means, that all scripts must be on the same level as app.py aka in the /app folder.

# Verwende ein offizielles Python-Image mit Version 3.11.0rc1 als Basis-Image
FROM python:3.11.0rc1

# Setze das Arbeitsverzeichnis im Container
WORKDIR /app

# Kopiere die Anforderungen in den Container
#Frage: Ist diese relative Pfadangabe möglich?
COPY requirements.txt .
#COPY kaggle.json: WARNING: THIS IS SIMONS PERSONAL KEY, REMOVE BEFORE DEPLOYMENT / PUBLISHING.
COPY kaggle.json /root/.kaggle/kaggle.json 

# Copy the current directory contents into the container at /app
COPY . /app

# Copy the data directory into the container at /app/data
#if this solves the import data.make_database import, proceed with all other modules in the same way
#COPY ../data /app/data

# Installiere die Abhängigkeiten
RUN pip install --no-cache-dir -r requirements.txt

# Kopiere den Rest des Anwendungscodes in den Container
#not necessary, if the directory is correctly copied to ./app
#COPY . .

#Expose the Port the Application is running on
EXPOSE 8000
# Starte die FastAPI-Anwendung mit Uvicorn
CMD ["uvicorn", "gateway_app:app", "--host", "0.0.0.0", "--port", "8000"]
