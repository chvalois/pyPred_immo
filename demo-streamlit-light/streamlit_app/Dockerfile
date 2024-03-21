FROM python:3.10-slim

# Install wget and unzip
RUN apt-get update && apt-get install -y curl unzip

WORKDIR /streamlit_app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY . .

# Download the ZIP file, unzip it there, and then delete the ZIP archive
RUN curl -fSL -o assets/file.zip "https://storage.cloud.google.com/pyprod_immo_databases/models.zip" && \
    unzip assets/file.zip -d assets && \
    rm assets/file.zip

CMD [ "streamlit", "run",  "app.py"]
