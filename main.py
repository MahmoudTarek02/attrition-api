# from fastapi import FastAPI
# from pydantic import BaseModel
# import joblib
# import pandas as pd

# # Import the class definition
# from attrition_preprocessor import AttritionPreprocessor

# # --------------------------
# # Load model + preprocessor
# # --------------------------

# model = joblib.load("best_attrition_model_xgboost.pkl")
# preprocessor: AttritionPreprocessor = joblib.load("attrition_preprocessor.pkl")

# # --------------------------
# # FastAPI App
# # --------------------------

# app = FastAPI(
#     title="Attrition Prediction API",
#     description="Send raw employee data and get attrition prediction",
#     version="1.0.0",
# )

# # --------------------------
# # Input schema (Pydantic)
# # --------------------------

# class Employee(BaseModel):
#     Age: int
#     Gender: str
#     Department: str
#     BusinessTravel: str
#     DailyRate: int
#     DistanceFromHome: int
#     Education: int
#     EducationField: str
#     EnvironmentSatisfaction: int
#     JobInvolvement: int
#     JobLevel: int
#     JobRole: str
#     JobSatisfaction: int
#     MaritalStatus: str
#     MonthlyIncome: int
#     NumCompaniesWorked: int
#     OverTime: str
#     PercentSalaryHike: int
#     PerformanceRating: int
#     RelationshipSatisfaction: int
#     StockOptionLevel: int
#     TotalWorkingYears: int
#     TrainingTimesLastYear: int
#     WorkLifeBalance: int
#     YearsAtCompany: int
#     YearsInCurrentRole: int
#     YearsSinceLastPromotion: int
#     YearsWithCurrManager: int

# # --------------------------
# # Prediction endpoint
# # --------------------------

# @app.post("/predict")
# def predict(data: Employee):
#     # Convert to dict → dataframe → preprocessed
#     raw = data.dict()
#     X = preprocessor.transform(raw)

#     # Model prediction
#     prob = model.predict_proba(X)[0][1]
#     pred = int(prob >= 0.5)

#     return {
#         "prediction": pred,         # 0 = Stay, 1 = Leave
#         "probability": float(prob)  # probability of leaving
#     }

# # --------------------------
# # Root endpoint
# # --------------------------

# @app.get("/")
# def root():
#     return {
#         "message": "Attrition Prediction API is running!",
#         "usage": "Send POST request to /predict with employee data."
#     }


from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import shap
import numpy as np
from typing import List


from attrition_preprocessor import AttritionPreprocessor

app = FastAPI()

# --------------------------
# Load model + preprocessor
# --------------------------
model = joblib.load("best_attrition_model_xgboost.pkl")
preprocessor: AttritionPreprocessor = joblib.load("attrition_preprocessor.pkl")

# Create SHAP explainer once
explainer = shap.Explainer(model)

# --------------------------
# Input Schema
# --------------------------


class Employee(BaseModel):
    EmployeeNumber: int
    Age: int
    Gender: str
    Department: str
    BusinessTravel: str
    DailyRate: int
    DistanceFromHome: int
    Education: int
    EducationField: str
    EnvironmentSatisfaction: int
    JobInvolvement: int
    JobLevel: int
    JobRole: str
    JobSatisfaction: int
    MaritalStatus: str
    MonthlyIncome: int
    NumCompaniesWorked: int
    OverTime: str
    PercentSalaryHike: int
    PerformanceRating: int
    RelationshipSatisfaction: int
    StockOptionLevel: int
    TotalWorkingYears: int
    TrainingTimesLastYear: int
    WorkLifeBalance: int
    YearsAtCompany: int
    YearsInCurrentRole: int
    YearsSinceLastPromotion: int
    YearsWithCurrManager: int

# --------------------------
# Prediction Endpoint
# --------------------------


@app.post("/predict")
def predict(employee: Employee):
    raw = employee.dict()

    emp_id = raw.get("EmployeeNumber", None)

    # Preprocess
    X = preprocessor.transform(raw)
    X_df = pd.DataFrame(X, columns=preprocessor.expected_features)

    # Predict
    proba = float(model.predict_proba(X)[0][1])
    pred = int(proba >= 0.5)

    # Risk level
    if proba < 0.3:
        risk = "Low"
    elif proba < 0.6:
        risk = "Medium"
    else:
        risk = "High"

    # SHAP Explanation
    shap_values = explainer(X)
    sv = shap_values.values[0]

    df_exp = pd.DataFrame({
        "feature": preprocessor.expected_features,
        "value": X[0],
        "shap": sv,
        "abs_shap": np.abs(sv)
    }).sort_values("abs_shap", ascending=False)

    # top 3
    top3 = df_exp.head(3)

    reasons = []
    for _, row in top3.iterrows():
        direction = "increases attrition risk" if row["shap"] > 0 else "reduces attrition risk"
        reasons.append(f"{row['feature']} ({direction})")

    return {
        "employee_number": emp_id,
        "prediction": pred,
        "probability": proba,
        "risk_level": risk,
        "reasons": reasons
    }
