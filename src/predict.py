import joblib
import numpy as np

model = joblib.load("models/model.pkl")

def predict(pclass, sex, age):
    data = np.array([[pclass, sex, age]])
    return model.predict(data)[0]