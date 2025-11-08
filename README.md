# Employee Attrition Prediction API

A FastAPI-based machine learning service that predicts employee attrition risk using IBM HR Analytics data. The API accepts raw employee data from a PostgreSQL database, applies the same preprocessing pipeline used during model training in Google Colab, and returns attrition predictions with risk levels.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Files Explanation](#files-explanation)
- [Setup & Installation](#setup--installation)
- [Running the Application](#running-the-application)
- [Testing the API](#testing-the-api)
- [n8n Workflow Integration](#n8n-workflow-integration)
- [API Endpoints](#api-endpoints)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Project Overview

### Problem Statement
The machine learning model was trained in Google Colab with preprocessed data (encoded categories, scaled features, 44 numerical features). However, the production database contains raw employee data (strings, categories, 31 fields). This API bridges that gap by applying the exact same preprocessing pipeline before making predictions.

### Solution
1. **Preprocessing Pipeline**: Replicates all preprocessing steps from Colab (label encoding, one-hot encoding, standard scaling)
2. **FastAPI Service**: Accepts raw employee data and returns predictions
3. **n8n Integration**: Automated workflow to fetch data from PostgreSQL, get predictions, and update the database

### What We Accomplished
✅ Trained multiple ML models in Google Colab (Logistic Regression, Random Forest, Gradient Boosting, XGBoost, SVM)  
✅ Selected best model (Gradient Boosting) and saved it  
✅ Created preprocessing pipeline matching Colab preprocessing  
✅ Built FastAPI service that accepts raw database data  
✅ Resolved Windows multiprocessing and NumPy compatibility issues  
✅ Integrated with n8n for automated batch predictions  
✅ Successfully predicted attrition for 10 sample employees and updated database  

---

## 🏗️ Architecture

```
PostgreSQL Database (Raw Data)
         ↓
    n8n Workflow
         ↓
FastAPI Service (/predict endpoint)
         ↓
Preprocessing Pipeline
    ├── Drop unnecessary columns
    ├── Label encode binary features (Gender, OverTime)
    ├── One-hot encode categorical features
    └── Standard scale all features
         ↓
Gradient Boosting Model
         ↓
Prediction Result (Attrition Risk)
         ↓
n8n Updates Database
```

---

## 📁 Files Explanation

### **Core Application Files** ✅ (Keep These)

#### `main.py` ⭐ **MAIN FILE**
- **Purpose**: FastAPI application with embedded preprocessing pipeline
- **Why**: Primary API service that handles all prediction requests
- **When Used**: Running the API server
- **Key Features**:
  - Embedded `AttritionPreprocessor` class (avoids pickle multiprocessing issues)
  - Lazy loading of model and preprocessor components
  - Handles NumPy version compatibility issues
  - REST API endpoints for predictions and health checks

#### `best_attrition_model_gradient_boosting.pkl`
- **Purpose**: Trained Gradient Boosting model from Google Colab
- **Why**: The ML model that makes attrition predictions
- **When Used**: Loaded when API starts, used for every prediction
- **Size**: ~1-5 MB
- **Training Details**: Trained on balanced data (SMOTE), tuned hyperparameters

#### `preprocessor_scaler.pkl`
- **Purpose**: Fitted StandardScaler from training data
- **Why**: Scales features to match training data distribution
- **When Used**: Loaded at API startup, applied to every prediction
- **Contains**: Mean and standard deviation for 44 features

#### `preprocessor_encoders.pkl`
- **Purpose**: Label encoders for binary categorical features
- **Why**: Encodes Gender (Male/Female) and OverTime (Yes/No) consistently
- **When Used**: Applied during preprocessing of each prediction
- **Contains**: Fitted LabelEncoder objects for 2 columns

#### `preprocessor_features.pkl`
- **Purpose**: List of 44 expected feature names in correct order
- **Why**: Ensures features are in the exact order the model expects
- **When Used**: During preprocessing to align feature columns
- **Contains**: Column names after one-hot encoding

#### `requirements.txt`
- **Purpose**: Python package dependencies
- **Why**: Ensures all required libraries are installed
- **When Used**: During initial setup (`pip install -r requirements.txt`)
- **Contents**:
  ```
  fastapi==0.104.1
  uvicorn[standard]==0.24.0
  pydantic==2.5.0
  scikit-learn==1.2.2
  numpy==1.23.5
  pandas==2.1.3
  joblib==1.3.2
  ```

---

### **Setup & Utility Files** ⚙️ (Keep for Reference/Maintenance)

#### `standalone_setup.py`
- **Purpose**: One-time setup script to create preprocessor components
- **Why**: Downloads dataset, fits preprocessor, saves components
- **When Used**: Initial setup or if preprocessor components need to be recreated
- **Run Once**: `python standalone_setup.py`
- **Output**: Creates preprocessor_scaler.pkl, preprocessor_encoders.pkl, preprocessor_features.pkl

#### `test_api.py`
- **Purpose**: Comprehensive test suite for API validation
- **Why**: Ensures all API functionality works correctly before deployment
- **When Used**: 
  - **Required**: After initial setup (before starting API)
  - After making changes to code
  - Before deployment
  - During debugging
- **How to Run**: `python test_api.py` (must run while API is running)
- **Tests**:
  - Health check endpoint
  - Feature info endpoint
  - Single employee prediction
  - Multiple employee predictions
  - Database simulation
- **Success Criteria**: All 5 tests must pass before using the API

#### `Dockerfile`
- **Purpose**: Docker containerization configuration
- **Why**: Enables deployment in containerized environments
- **When Used**: Docker deployment (optional)
- **Run**: `docker build -t attrition-api . && docker run -p 8000:8000 attrition-api`

---

## 🗄️ Database Setup

### PostgreSQL Table Schema

Create the table in your PostgreSQL database to store employee data and predictions:

```sql
CREATE TABLE employee_data (
    id SERIAL PRIMARY KEY,
    EmployeeNumber INTEGER,
    Age INTEGER,
    BusinessTravel VARCHAR(50),
    DailyRate INTEGER,
    Department VARCHAR(100),
    DistanceFromHome INTEGER,
    Education INTEGER,
    EducationField VARCHAR(100),
    EnvironmentSatisfaction INTEGER,
    Gender VARCHAR(20),
    HourlyRate INTEGER,
    JobInvolvement INTEGER,
    JobLevel INTEGER,
    JobRole VARCHAR(100),
    JobSatisfaction INTEGER,
    MaritalStatus VARCHAR(50),
    MonthlyIncome INTEGER,
    MonthlyRate INTEGER,
    NumCompaniesWorked INTEGER,
    OverTime VARCHAR(10),
    PercentSalaryHike INTEGER,
    PerformanceRating INTEGER,
    RelationshipSatisfaction INTEGER,
    StockOptionLevel INTEGER,
    TotalWorkingYears INTEGER,
    TrainingTimesLastYear INTEGER,
    WorkLifeBalance INTEGER,
    YearsAtCompany INTEGER,
    YearsInCurrentRole INTEGER,
    YearsSinceLastPromotion INTEGER,
    YearsWithCurrManager INTEGER,
    
    -- Prediction columns (populated by API)
    attrition_prediction INTEGER,
    attrition_probability DECIMAL(5,4),
    risk_level VARCHAR(20),
    prediction_date TIMESTAMP,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Insert Sample Data (10 Test Employees)

```sql
INSERT INTO employee_data (
    EmployeeNumber, Age, BusinessTravel, DailyRate, Department, DistanceFromHome,
    Education, EducationField, EnvironmentSatisfaction, Gender, HourlyRate,
    JobInvolvement, JobLevel, JobRole, JobSatisfaction, MaritalStatus, MonthlyIncome,
    MonthlyRate, NumCompaniesWorked, OverTime, PercentSalaryHike, PerformanceRating,
    RelationshipSatisfaction, StockOptionLevel, TotalWorkingYears, TrainingTimesLastYear,
    WorkLifeBalance, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion,
    YearsWithCurrManager
) VALUES
(1, 41, 'Travel_Rarely', 1102, 'Sales', 1,
 2, 'Life Sciences', 2, 'Female', 94,
 3, 2, 'Sales Executive', 4, 'Single', 5993,
 19479, 8, 'Yes', 11, 3,
 1, 0, 8, 0,
 1, 6, 4, 0, 5),
(2, 34, 'Travel_Rarely', 900, 'Research & Development', 3,
 2, 'Life Sciences', 3, 'Male', 65,
 3, 2, 'Research Scientist', 3, 'Married', 4800,
 18000, 3, 'No', 12, 3,
 3, 1, 6, 2,
 3, 4, 3, 3, 2),
(3, 29, 'Travel_Frequently', 1200, 'Sales', 6,
 3, 'Marketing', 4, 'Female', 70,
 2, 2, 'Sales Executive', 2, 'Single', 5200,
 21000, 2, 'Yes', 13, 3,
 2, 0, 4, 3,
 2, 4, 2, 1, 1),
(4, 45, 'Non-Travel', 800, 'Human Resources', 1,
 1, 'Human Resources', 2, 'Female', 55,
 3, 3, 'Human Resources', 4, 'Married', 6400,
 23000, 1, 'No', 11, 3,
 3, 0, 20, 3,
 2, 10, 7, 3, 4),
(5, 38, 'Travel_Rarely', 1100, 'Sales', 10,
 4, 'Medical', 2, 'Male', 67,
 2, 3, 'Sales Executive', 4, 'Divorced', 6200,
 19500, 5, 'No', 15, 4,
 3, 1, 12, 2,
 3, 8, 6, 2, 5),
(6, 50, 'Non-Travel', 700, 'Research & Development', 4,
 5, 'Technical Degree', 1, 'Male', 60,
 4, 4, 'Manager', 2, 'Married', 9800,
 26000, 4, 'Yes', 18, 4,
 1, 2, 25, 4,
 2, 15, 10, 4, 6),
(7, 27, 'Travel_Frequently', 1300, 'Sales', 8,
 1, 'Marketing', 3, 'Female', 75,
 3, 2, 'Sales Representative', 3, 'Single', 3500,
 17000, 2, 'Yes', 14, 3,
 4, 1, 4, 2,
 3, 3, 2, 1, 2),
(8, 32, 'Travel_Rarely', 1000, 'Research & Development', 2,
 3, 'Life Sciences', 4, 'Female', 62,
 4, 3, 'Research Scientist', 4, 'Married', 5700,
 20000, 3, 'No', 12, 3,
 4, 0, 7, 2,
 4, 7, 5, 2, 3),
(9, 36, 'Travel_Rarely', 1050, 'Research & Development', 3,
 4, 'Medical', 2, 'Male', 68,
 3, 2, 'Laboratory Technician', 3, 'Divorced', 4500,
 18500, 4, 'No', 13, 3,
 3, 1, 6, 1,
 3, 5, 4, 3, 2),
(10, 43, 'Non-Travel', 950, 'Human Resources', 4,
 2, 'Human Resources', 1, 'Female', 57,
 2, 3, 'Human Resources', 2, 'Married', 5400,
 19000, 3, 'No', 11, 3,
 2, 0, 18, 4,
 3, 12, 10, 4, 5);
```

### Verify Data Inserted

```sql
SELECT EmployeeNumber, Age, Department, JobRole, Gender, OverTime
FROM employee_data
ORDER BY EmployeeNumber;
```

Expected: 10 rows returned

---

### **Files You Can Delete** ❌

The following files are no longer needed and can be safely deleted:

```bash
rm app.py setup_preprocessor_components.py
rm -rf __pycache__
```

- **`app.py`**: Old version of API, replaced by `main.py`
- **`setup_preprocessor_components.py`**: Old setup script, replaced by `standalone_setup.py`
- **`__pycache__/`**: Auto-generated Python cache (will be recreated automatically)

**Note**: Keep `venv/` or conda environment, but don't commit to git (add to `.gitignore`)

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.8 - 3.11 (Python 3.12 may have compatibility issues)
- PostgreSQL database with employee data
- n8n instance (for automation)

### Step 1: Clone/Download Project
```bash
cd D:\attrition_api
```

### Step 2: Create Conda Environment
```bash
# Create conda environment
conda create -n attrition_env python=3.10 -y

# Activate environment
conda activate attrition_env
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Important**: If you encounter NumPy compatibility issues:
```bash
pip install numpy==1.23.5 scikit-learn==1.2.2
```

### Step 4: Verify Preprocessor Components (Optional)
If preprocessor files don't exist, run:
```bash
python standalone_setup.py
```

This will:
- Download IBM HR Attrition dataset from Kaggle
- Fit the preprocessor on the dataset
- Save all component files

### Step 5: Verify Model File
Ensure `best_attrition_model_gradient_boosting.pkl` exists in the project directory.

---

## 🏃 Running the Application

### Prerequisites Check
Before starting the API, ensure:
- ✅ Conda environment is activated (`conda activate attrition_env`)
- ✅ All dependencies installed (`pip install -r requirements.txt`)
- ✅ All `.pkl` files exist in the directory
- ✅ PostgreSQL database is set up with sample data (optional for testing)

### Start the FastAPI Server

**Option 1: Development Mode (with auto-reload)**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Option 2: Production Mode**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

**Option 3: Using Docker**
```bash
docker build -t attrition-api .
docker run -p 8000:8000 attrition-api
```

### Verify API is Running
Open your browser and go to:
- **API Root**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

You should see:
```json
{
  "message": "Employee Attrition Prediction API v2.0",
  "status": "active",
  "endpoints": {
    "predict": "/predict (POST)",
    "health": "/health (GET)",
    "docs": "/docs"
  }
}
```

---

## 🧪 Testing the API

### Step 1: Start the API Server (Required First!)

**The API must be running before you can test it.**

Open a terminal and start the server:
```bash
conda activate attrition_env
uvicorn main:app --reload
```

Keep this terminal open and running.

---

### Step 2: Run Comprehensive Test Suite

**Open a NEW terminal** (keep the API server running in the first one) and run:

```bash
conda activate attrition_env
python test_api.py
```

**What each test validates:**

1. **TEST 1: Health Check**
   - **Purpose**: Verifies the API server is running and healthy
   - **What it checks**: 
     - API responds to `/health` endpoint
     - Model is loaded successfully
     - Preprocessor is loaded successfully
     - Expected features count (44) is correct
   - **Why it matters**: Ensures basic API functionality before attempting predictions

2. **TEST 2: Feature Info**
   - **Purpose**: Validates preprocessing pipeline configuration
   - **What it checks**:
     - Total features after preprocessing (should be 44)
     - Preprocessing steps are correctly defined
     - Binary and one-hot encoded columns are listed
   - **Why it matters**: Confirms the preprocessing pipeline matches training configuration

3. **TEST 3: Raw Data Prediction (Single Employee)**
   - **Purpose**: Tests end-to-end prediction with one employee
   - **What it checks**:
     - API accepts raw employee data (31 fields)
     - Preprocessing transforms data correctly (31 → 44 features)
     - Model makes prediction successfully
     - Returns prediction, probability, risk level, and message
   - **Why it matters**: Validates the core functionality - raw data in, prediction out
   - **Test Data**: Female Sales Executive, 41 years old, works overtime

4. **TEST 4: Multiple Employees**
   - **Purpose**: Tests batch prediction capability
   - **What it checks**:
     - API handles multiple sequential predictions
     - Different employee profiles produce different results
     - Response format is consistent across predictions
   - **Why it matters**: Simulates n8n workflow processing multiple employees
   - **Test Data**: 2 employees with contrasting profiles (low risk vs high risk)

5. **TEST 5: Database Simulation**
   - **Purpose**: Simulates real-world database integration
   - **What it checks**:
     - Data from "database query" is processed correctly
     - Field names match database schema
     - Prediction completes successfully
   - **Why it matters**: Validates the integration path: PostgreSQL → n8n → API
   - **Test Data**: Simulates a SELECT query result for Employee #12345

**Expected output:**
```
================================================================================
ATTRITION API - COMPREHENSIVE TEST SUITE
================================================================================

================================================================================
TEST 1: Health Check
================================================================================
Status Code: 200
Response: {
  "status": "healthy",
  "model_loaded": true,
  "preprocessor_loaded": true,
  "expected_features": 44
}

...

================================================================================
TEST SUMMARY
================================================================================
Health Check                   ✓ PASSED
Feature Info                   ✓ PASSED
Raw Data Prediction           ✓ PASSED
Multiple Employees            ✓ PASSED
Database Simulation           ✓ PASSED

Total: 5/5 tests passed

🎉 All tests passed! Your API is ready for production.
```

**If tests fail:**
- ❌ **"Could not connect to API"**: Make sure FastAPI is running in another terminal
- ❌ **Test failures**: Check error messages and see [Troubleshooting](#troubleshooting) section
- ❌ **Import errors**: Verify all packages are installed (`pip install -r requirements.txt`)

---

### Step 3: Quick Manual Tests

With the API still running, you can also test manually:

#### Quick Health Check
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "preprocessor_loaded": true,
  "expected_features": 44
}
```

### Test Prediction with Sample Data
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 41,
    "BusinessTravel": "Travel_Rarely",
    "DailyRate": 1102,
    "Department": "Sales",
    "DistanceFromHome": 1,
    "Education": 2,
    "EducationField": "Life Sciences",
    "EmployeeNumber": 1,
    "EnvironmentSatisfaction": 2,
    "Gender": "Female",
    "HourlyRate": 94,
    "JobInvolvement": 3,
    "JobLevel": 2,
    "JobRole": "Sales Executive",
    "JobSatisfaction": 4,
    "MaritalStatus": "Single",
    "MonthlyIncome": 5993,
    "MonthlyRate": 19479,
    "NumCompaniesWorked": 8,
    "OverTime": "Yes",
    "PercentSalaryHike": 11,
    "PerformanceRating": 3,
    "RelationshipSatisfaction": 1,
    "StockOptionLevel": 0,
    "TotalWorkingYears": 8,
    "TrainingTimesLastYear": 0,
    "WorkLifeBalance": 1,
    "YearsAtCompany": 6,
    "YearsInCurrentRole": 4,
    "YearsSinceLastPromotion": 0,
    "YearsWithCurrManager": 5
  }'
```

Expected response:
```json
{
  "employee_number": 1,
  "attrition_prediction": 1,
  "attrition_probability": 0.7234,
  "risk_level": "High",
  "message": "Employee is predicted to leave. Risk level: High"
}
```

---

### Summary: Correct Testing Order

1. **First**: `uvicorn main:app --reload` (Terminal 1 - keep running)
2. **Then**: `python test_api.py` (Terminal 2 - run tests)
3. **Finally**: Manual tests with curl or browser (optional)

**Important**: The API server must remain running for all tests to work!

---

---

### Run Comprehensive Test Suite (Automated)

**Prerequisites**: FastAPI server must be running in another terminal!

```bash
python test_api.py
```

This tests:
- ✅ Health check endpoint
- ✅ Feature info endpoint  
- ✅ Single prediction
- ✅ Multiple predictions
- ✅ Database simulation

**Note**: This script makes actual HTTP requests to http://localhost:8000, so the server must be running first!

---

## 🔄 n8n Workflow Integration

### Workflow Overview
The n8n workflow automates the entire prediction pipeline:

1. **Fetch Data**: Query PostgreSQL for employees needing predictions
2. **Loop Through Records**: Process each employee individually
3. **Call API**: POST to `/predict` endpoint with employee data
4. **Update Database**: Write prediction results back to PostgreSQL

### Prerequisites
- ✅ FastAPI server running (`uvicorn main:app --reload`)
- ✅ PostgreSQL database with employee_data table
- ✅ n8n instance accessible
- ✅ Sample data inserted (10 employees)

### n8n Workflow Steps

#### Step 1: PostgreSQL - Fetch Employees

**Node Type**: Postgres  
**Operation**: Execute Query  
**Query**:

```sql
SELECT 
  EmployeeNumber,
  Age,
  BusinessTravel,
  DailyRate,
  Department,
  DistanceFromHome,
  Education,
  EducationField,
  EnvironmentSatisfaction,
  Gender,
  HourlyRate,
  JobInvolvement,
  JobLevel,
  JobRole,
  JobSatisfaction,
  MaritalStatus,
  MonthlyIncome,
  MonthlyRate,
  NumCompaniesWorked,
  OverTime,
  PercentSalaryHike,
  PerformanceRating,
  RelationshipSatisfaction,
  StockOptionLevel,
  TotalWorkingYears,
  TrainingTimesLastYear,
  WorkLifeBalance,
  YearsAtCompany,
  YearsInCurrentRole,
  YearsSinceLastPromotion,
  YearsWithCurrManager
FROM employee_data
WHERE attrition_prediction IS NULL
LIMIT 10;
```

**Purpose**: Fetches employees who haven't received predictions yet

#### Step 2: Loop Over Items
- **Node**: Loop Over Items
- **Batch Size**: 1 (process one employee at a time)

#### Step 3: HTTP Request - Predict

**Node Type**: HTTP Request  
**Method**: POST  
**URL**: `http://localhost:8000/predict`  
**Headers**: 
  - `Content-Type: application/json`
  
**Body** (JSON):
```json
{
  "Age": "{{ $json.Age }}",
  "BusinessTravel": "{{ $json.BusinessTravel }}",
  "DailyRate": "{{ $json.DailyRate }}",
  "Department": "{{ $json.Department }}",
  "DistanceFromHome": "{{ $json.DistanceFromHome }}",
  "Education": "{{ $json.Education }}",
  "EducationField": "{{ $json.EducationField }}",
  "EmployeeNumber": "{{ $json.EmployeeNumber }}",
  "EnvironmentSatisfaction": "{{ $json.EnvironmentSatisfaction }}",
  "Gender": "{{ $json.Gender }}",
  "HourlyRate": "{{ $json.HourlyRate }}",
  "JobInvolvement": "{{ $json.JobInvolvement }}",
  "JobLevel": "{{ $json.JobLevel }}",
  "JobRole": "{{ $json.JobRole }}",
  "JobSatisfaction": "{{ $json.JobSatisfaction }}",
  "MaritalStatus": "{{ $json.MaritalStatus }}",
  "MonthlyIncome": "{{ $json.MonthlyIncome }}",
  "MonthlyRate": "{{ $json.MonthlyRate }}",
  "NumCompaniesWorked": "{{ $json.NumCompaniesWorked }}",
  "OverTime": "{{ $json.OverTime }}",
  "PercentSalaryHike": "{{ $json.PercentSalaryHike }}",
  "PerformanceRating": "{{ $json.PerformanceRating }}",
  "RelationshipSatisfaction": "{{ $json.RelationshipSatisfaction }}",
  "StockOptionLevel": "{{ $json.StockOptionLevel }}",
  "TotalWorkingYears": "{{ $json.TotalWorkingYears }}",
  "TrainingTimesLastYear": "{{ $json.TrainingTimesLastYear }}",
  "WorkLifeBalance": "{{ $json.WorkLifeBalance }}",
  "YearsAtCompany": "{{ $json.YearsAtCompany }}",
  "YearsInCurrentRole": "{{ $json.YearsInCurrentRole }}",
  "YearsSinceLastPromotion": "{{ $json.YearsSinceLastPromotion }}",
  "YearsWithCurrManager": "{{ $json.YearsWithCurrManager }}"
}
```

**Purpose**: Calls FastAPI to get attrition prediction for each employee

#### Step 4: PostgreSQL - Update Results

**Node Type**: Postgres  
**Operation**: Execute Query  
**Query**:

```sql
UPDATE employee_data
SET 
  attrition_prediction = {{ $json.attrition_prediction }},
  attrition_probability = {{ $json.attrition_probability }},
  risk_level = '{{ $json.risk_level }}',
  prediction_date = NOW()
WHERE EmployeeNumber = {{ $json.employee_number }};
```

**Purpose**: Writes prediction results back to the database

### Running the n8n Workflow

#### Step-by-Step Execution:

1. **Start FastAPI Server** (if not already running)
   ```bash
   conda activate attrition_env
   uvicorn main:app --reload
   ```

2. **Verify API is Running**
   ```bash
   curl http://localhost:8000/health
   ```
   Should return: `{"status": "healthy", ...}`

3. **Open n8n Workflow**
   - Navigate to your n8n instance
   - Create new workflow or import existing

4. **Configure PostgreSQL Node**
   - Add PostgreSQL credentials
   - Test connection
   - Set the fetch query (Step 1 above)

5. **Configure HTTP Request Node**
   - Set URL: `http://localhost:8000/predict`
   - Set method: POST
   - Add body mapping (Step 3 above)
   - Test with single execution

6. **Configure Update Node**
   - Use same PostgreSQL credentials
   - Set update query (Step 4 above)

7. **Execute Workflow**
   - Click "Execute Workflow" button
   - Monitor execution in real-time
   - Check for errors in each node

8. **Verify Results in Database**
   ```sql
   SELECT 
     EmployeeNumber,
     Age,
     Department,
     attrition_prediction,
     attrition_probability,
     risk_level,
     prediction_date
   FROM employee_data
   WHERE attrition_prediction IS NOT NULL
   ORDER BY EmployeeNumber;
   ```

### Expected Results
After running the workflow on 10 sample employees:
- ✅ All 10 employees have predictions
- ✅ Database updated with `attrition_prediction` (0 or 1)
- ✅ Database updated with `attrition_probability` (0.0 - 1.0)
- ✅ Database updated with `risk_level` (Low/Medium/High)
- ✅ Timestamp in `prediction_date` column

---

## 🔌 API Endpoints

### `GET /`
**Description**: API root, returns status and available endpoints

**Response**:
```json
{
  "message": "Employee Attrition Prediction API v2.0",
  "status": "active",
  "endpoints": {...}
}
```

---

### `GET /health`
**Description**: Health check endpoint

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "preprocessor_loaded": true,
  "expected_features": 44
}
```

---

### `POST /predict`
**Description**: Predict employee attrition from raw data

**Request Body**:
```json
{
  "Age": 41,
  "BusinessTravel": "Travel_Rarely",
  "Department": "Sales",
  "Gender": "Female",
  "OverTime": "Yes",
  "JobRole": "Sales Executive",
  ...
}
```

**Response**:
```json
{
  "employee_number": 1,
  "attrition_prediction": 1,
  "attrition_probability": 0.7234,
  "risk_level": "High",
  "message": "Employee is predicted to leave. Risk level: High"
}
```

**Valid Categorical Values**:
- **BusinessTravel**: "Travel_Rarely", "Travel_Frequently", "Non-Travel"
- **Department**: "Sales", "Research & Development", "Human Resources"
- **EducationField**: "Life Sciences", "Medical", "Marketing", "Technical Degree", "Other", "Human Resources"
- **Gender**: "Male", "Female"
- **JobRole**: "Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"
- **MaritalStatus**: "Single", "Married", "Divorced"
- **OverTime**: "Yes", "No"

---

### `GET /feature-info`
**Description**: Get preprocessing details and expected features

**Response**:
```json
{
  "total_features_after_preprocessing": 44,
  "preprocessing_steps": [...],
  "binary_encoded_columns": ["Gender", "OverTime"],
  "one_hot_encoded_columns": [...],
  "sample_features": [...]
}
```

---

## 🔧 Troubleshooting

### Issue: "Can't get attribute 'AttritionPreprocessor'"
**Cause**: Pickle multiprocessing issue on Windows  
**Solution**: Use `main.py` (has embedded preprocessor class)

### Issue: "NumPy BitGenerator error"
**Cause**: NumPy version mismatch between Colab and local  
**Solution**: 
```bash
pip install numpy==1.23.5 scikit-learn==1.2.2
```

### Issue: "Model file not found"
**Cause**: Model file missing from directory  
**Solution**: Download from Colab and place in project root

### Issue: "y contains previously unseen labels"
**Cause**: Invalid categorical value in request  
**Solution**: Use valid values (see API Endpoints section)

### Issue: n8n workflow fails
**Cause**: FastAPI not running or wrong URL  
**Solution**: 
1. Verify API is running: `curl http://localhost:8000/health`
2. Check n8n HTTP node URL is correct
3. Ensure field mapping matches database columns

---

## 📊 Preprocessing Pipeline Details

The API applies the **exact same preprocessing** as Google Colab training:

### Step 1: Drop Columns
Removes constant and identifier columns:
- `EmployeeCount` (always 1)
- `StandardHours` (always 80)
- `Over18` (always Y)
- `EmployeeNumber` (identifier)

### Step 2: Label Encoding (Binary Features)
Converts binary categories to 0/1:
- `Gender`: Male=0, Female=1
- `OverTime`: No=0, Yes=1

### Step 3: One-Hot Encoding (Categorical Features)
Creates binary columns for each category (drop_first=True):
- `BusinessTravel` → BusinessTravel_Travel_Frequently, BusinessTravel_Travel_Rarely
- `Department` → Department_Research & Development, Department_Sales
- `EducationField` → Multiple columns
- `JobRole` → Multiple columns
- `MaritalStatus` → MaritalStatus_Married, MaritalStatus_Single

### Step 4: Standard Scaling
Scales all 44 features using fitted StandardScaler:
- Mean = 0, Standard Deviation = 1
- Uses statistics from training data

**Result**: 31 raw fields → 44 preprocessed numerical features

---

## 📈 Model Performance

**Model**: Gradient Boosting Classifier  
**Training Data**: IBM HR Analytics Attrition Dataset (1,470 employees)  
**Class Balance**: SMOTE applied (0.5 ratio)  
**Test Set Performance**:
- ROC-AUC: ~0.85
- F1-Score: ~0.75
- Accuracy: ~0.82

**Risk Levels**:
- **Low**: Probability < 0.3 (unlikely to leave)
- **Medium**: Probability 0.3 - 0.6 (moderate risk)
- **High**: Probability > 0.6 (likely to leave)

---

## 🎯 Production Checklist

Before deploying to production:

- [ ] All tests pass (`python test_api.py`)
- [ ] Health endpoint returns "healthy"
- [ ] n8n workflow tested with sample data
- [ ] Database columns exist (attrition_prediction, attrition_probability, risk_level)
- [ ] API accessible from n8n server
- [ ] Error handling tested (invalid input, missing fields)
- [ ] Logging configured
- [ ] Consider API rate limiting
- [ ] Consider authentication (API keys)
- [ ] Monitor API performance

---

## 📝 License & Credits

**Dataset**: IBM HR Analytics Employee Attrition Dataset (Kaggle)  
**Framework**: FastAPI, scikit-learn  
**Automation**: n8n  
**Database**: PostgreSQL

---

## 🤝 Support

For issues or questions:
1. Check this README
2. Review test_api.py for examples
3. Check FastAPI interactive docs: http://localhost:8000/docs

---

**Last Updated**: November 2025  
**Version**: 2.0.0  
**Status**: ✅ Production Ready