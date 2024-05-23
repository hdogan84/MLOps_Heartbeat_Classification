# Verwende ein offizielles Python-Image mit Version 3.11.0rc1 als Basis-Image
FROM python:3.11.0rc1

# Setze das Arbeitsverzeichnis im Container
WORKDIR /app

# Kopiere die Anforderungen in den Container
#Frage: Ist diese relative Pfadangabe möglich?
COPY ../../requirements.txt .

# Installiere die Abhängigkeiten
RUN pip install --no-cache-dir -r requirements.txt

# Kopiere den Rest des Anwendungscodes in den Container
COPY . .

# Starte die FastAPI-Anwendung mit Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
