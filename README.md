# Employee Attrition Prediction API (GCP Deployment)
## Overview

- This repository hosts a predictive machine learning model trained on the IBM Employee Attrition Dataset.
- It is deployed on Google Cloud Platform (GCP) and exposes a public REST API endpoint for use by the Backend team.

## Input Format (Request Body)

### The model expects a JSON payload in the following format:
{
  "features": [
    41, 1102, 1, 2, 2, 0, 94, 3, 2, 4, 5993, 19479, 8, 1, 11, 3, 1, 0, 8, 0, 1, 6, 4, 0, 5,
    false, true, false, true, true, false, false, false, false, false, false, false,
    false, false, false, true, false, false, true
  ]
}
### Each number (or boolean) represents one feature value for a single employee record.

### Feature Names (After Preprocessing)

The model expects 44 features, in the following order:
['Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
 'Gender', 'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction',
 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'OverTime',
 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
 'YearsSinceLastPromotion', 'YearsWithCurrManager',
 'BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely',
 'Department_Research & Development', 'Department_Sales',
 'EducationField_Life Sciences', 'EducationField_Marketing',
 'EducationField_Medical', 'EducationField_Other',
 'EducationField_Technical Degree', 'JobRole_Human Resources',
 'JobRole_Laboratory Technician', 'JobRole_Manager',
 'JobRole_Manufacturing Director', 'JobRole_Research Director',
 'JobRole_Research Scientist', 'JobRole_Sales Executive',
 'JobRole_Sales Representative', 'MaritalStatus_Married',
 'MaritalStatus_Single']

## Example Output
{
  "prediction": "Yes",
  "probability": 0.83
}

# Running the API Locally

## 1. Install dependencies

pip install -r requirements.txt


## 2. Start the API server

uvicorn app:app --host 0.0.0.0 --port 8080 --reload


## 3. Open browser on:

http://localhost:8080

## 4. Once running, you can test the API directly from:

Swagger UI: http://localhost:8080/docs

## 5. API Endpoints

### 5.1 Health check endpoint: Health check endpoint: 

Endpoint: GET /

Response: {"message": "Attrition Prediction API is live!"}

### 5.2 Predict employee attrition probability:

Endpoint: POST /predict

Request Body:

{
  "features": [
    41, 1102, 1, 2, 2, 0, 94, 3, 2, 4, 5993, 19479, 8, 1, 11, 3, 1, 0, 8, 0, 1, 6, 4, 0, 5,
    false, true, false, true, true, false, false, false, false, false, false, false,
    false, false, false, true, false, false, true
  ]
}

Response: 

{
  "attrition_prediction": 1,
  "attrition_probability": 0.8321
}





