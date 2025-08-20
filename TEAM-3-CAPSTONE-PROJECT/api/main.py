# api/main.py
from fastapi import FastAPI, HTTPException, Request
from api.schemas import HeartFeatures
import joblib, time, logging, os, pandas as pd
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

app = FastAPI(title="Heart Disease Predictor", version="1.0.0")

# templates
templates = Jinja2Templates(directory="templates")

MODEL_PATH = os.getenv("MODEL_PATH", "models/heart_pipeline.pkl")
bundle = joblib.load(MODEL_PATH)
pipe = bundle["pipeline"]
feature_order = bundle["features"]

os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename="logs/predictions.log", level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

@app.get("/", response_class=HTMLResponse)
def ui(request: Request):
    # renders the responsive HTML UI
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
def health():
    return {"status": "ok", "model_path": MODEL_PATH}

@app.post("/predict")
def predict(payload: HeartFeatures):
    try:
        t0 = time.time()
        X = pd.DataFrame([payload.dict()], columns=feature_order)
        prob = float(pipe.predict_proba(X)[0, 1])
        pred = int(prob >= 0.5)
        latency_ms = int((time.time() - t0) * 1000)
        logging.info({"features": payload.dict(), "prediction": pred, "prob": prob, "latency_ms": latency_ms})
        return {"prediction": pred, "probability": prob, "latency_ms": latency_ms}
    except Exception as e:
        logging.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))
