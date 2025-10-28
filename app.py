from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Attrition Prediction API")

model = joblib.load("best_attrition_model_gradient_boosting.pkl")

class EmployeeData(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(data: EmployeeData):
    try:
        print("Received data:", data.features)   # debug log
        X = np.array(data.features).reshape(1, -1)
        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[0][1]
        return {
            "attrition_prediction": int(prediction),
            "attrition_probability": round(float(proba), 4)
        }
    except Exception as e:
        print("Error:", e)   # debug log
        return {"error": str(e)}


@app.get("/")
def root():
    return {"message": "Attrition Prediction API is live!"}
