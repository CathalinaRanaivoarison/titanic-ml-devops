import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

from src.model import get_model
from src.preprocess import load_and_preprocess

# -------------------------
# Reproducibility
# -------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# -------------------------
# Load & preprocess data
# -------------------------
df = load_and_preprocess("data/train.csv")

# -------------------------
# Split
# -------------------------
X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# -------------------------
# Model
# -------------------------
model = get_model()
model.fit(X_train, y_train)

# -------------------------
# Evaluation
# -------------------------
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

print(f"Accuracy: {acc:.4f}")

# -------------------------
# Save model
# -------------------------
joblib.dump(model, "models/model.pkl")