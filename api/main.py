from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("models/model.pkl")

@app.post("/predict")
def predict(pclass: int, sex: int, age: float):
    data = np.array([[pclass, sex, age]])
    prediction = model.predict(data)[0]
    return {"prediction": int(prediction)}