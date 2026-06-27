import os
import joblib


MODEL_PATH = os.path.join("app", "model.pkl")


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model file not found. Please run train.py before starting the API."
        )
    return joblib.load(MODEL_PATH)