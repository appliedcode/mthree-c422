import mlflow.pyfunc
from pydantic import BaseModel, Field
from typing import List, Annotated
from fastapi import FastAPI
from pydantic import  conlist

# Must match the feature order as in training!
FEATURE_NAMES = [
    "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
    "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error", "smoothness error",
    "compactness error", "concavity error", "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area", "worst smoothness",
    "worst compactness", "worst concavity", "worst concave points", "worst symmetry", "worst fractal dimension"
]

class CancerInput(BaseModel):
    features: Annotated[List[float], Field(min_length=30, max_length=30)]

app = FastAPI()

@app.on_event("startup")
def load_model():
    # Loads the latest Production model. Change "Production" to "Staging" if needed.
    global model
    model = mlflow.pyfunc.load_model("models:/breast_cancer_diagnosis@production")

@app.get("/health")
def health():
    return {"status": "OK"}

@app.post("/predict")
def predict(inp: CancerInput):
    preds = model.predict([inp.features])
    prob = model.predict_proba([inp.features])[:,1].tolist()[0]
    label = int(preds)
    return {
        "prediction": label,       # 0=benign, 1=malignant
        "probability_malignant": prob
    }