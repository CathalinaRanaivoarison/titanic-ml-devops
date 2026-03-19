# Titanic ML DevOps Project

Simple ML project to predict Titanic survival using a RandomForestClassifier, with a FastAPI inference API and Docker deployment.

## Train

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/train.py
```

## Run API

```bash
uvicorn api.main:app --reload
# then open http://127.0.0.1:8000/docs
```

## Docker

```bash
docker build -t titanic-api .
docker run -p 8000:8000 titanic-api
```