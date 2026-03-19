# Titanic ML DevOps Project

ML project predicting Titanic survival using RandomForestClassifier, with FastAPI inference API and Docker deployment.

## Stack
- Python 3.10, scikit-learn, FastAPI, Docker
- Dataset: Titanic (Kaggle) - 891 samples

## Train model
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/train.py
```

## Run API locally
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
# Open http://127.0.0.1:8000/docs
```

## Docker deployment
```bash
docker build -t titanic-api .
docker run -p 8000:8000 titanic-api
```

Test endpoint:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"pclass": 3, "sex": 1, "age": 25}'
```