from fastapi import FastAPI
from app.schemas import IrisInput
from app.model_utils import load_model

app = FastAPI(title="Advanced Testing Pipeline API")

model = load_model()


@app.get("/")
def home():
    return {"message": "Advanced Testing Pipeline API is running"}


@app.post("/predict")
def predict(data: IrisInput):
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = model.predict(features)[0]

    return {
        "prediction": int(prediction)
    }