from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Optional, Dict, Any
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Employee Attrition Prediction API",
    description="Predicts employee attrition probability using raw employee data",
    version="2.0.0"
)

# ============================================================================
# PREPROCESSOR CLASS - EMBEDDED (NO PICKLE LOADING ISSUES)
# ============================================================================

class AttritionPreprocessor:
    """
    Preprocessing pipeline for IBM HR Attrition dataset.
    Must match the exact preprocessing steps used during training.
    """
    
    def __init__(self):
        # Columns to drop (constant + identifier columns)
        self.cols_to_drop = ['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber']
        
        # Binary categorical columns for Label Encoding
        self.binary_cols = ['Gender', 'OverTime']
        
        # Multi-class categorical columns for One-Hot Encoding
        self.multi_class_cols = ['BusinessTravel', 'Department', 'EducationField', 
                                  'JobRole', 'MaritalStatus']
        
        # Label encoders for binary columns
        self.label_encoders = {}
        
        # Standard scaler for numerical features
        self.scaler = StandardScaler()
        
        # Expected feature columns after preprocessing (in order)
        self.expected_features = None
        
    def fit(self, df: pd.DataFrame):
        """Fit the preprocessor on training data."""
        df_clean = df.copy()
        
        # Step 1: Drop unnecessary columns
        df_clean = df_clean.drop(columns=self.cols_to_drop, errors='ignore')
        
        # Step 2: Encode target variable (Attrition)
        if 'Attrition' in df_clean.columns:
            df_clean['Attrition'] = df_clean['Attrition'].map({'Yes': 1, 'No': 0})
        
        # Step 3: Label Encoding for binary columns
        for col in self.binary_cols:
            if col in df_clean.columns:
                le = LabelEncoder()
                df_clean[col] = le.fit_transform(df_clean[col])
                self.label_encoders[col] = le
        
        # Step 4: One-Hot Encoding for multi-class columns
        df_clean = pd.get_dummies(df_clean, columns=self.multi_class_cols, drop_first=True)
        
        # Step 5: Separate features and target
        if 'Attrition' in df_clean.columns:
            X = df_clean.drop('Attrition', axis=1)
        else:
            X = df_clean
        
        # Store expected feature columns
        self.expected_features = X.columns.tolist()
        
        # Step 6: Fit scaler
        self.scaler.fit(X)
        
        return self
    
    def transform(self, data: Dict[str, Any]) -> np.ndarray:
        """Transform raw employee data to preprocessed features."""
        # Convert dict to DataFrame
        df = pd.DataFrame([data])
        
        # Step 1: Drop unnecessary columns
        df_clean = df.drop(columns=self.cols_to_drop, errors='ignore')
        
        # Step 2: Encode target variable if present
        if 'Attrition' in df_clean.columns:
            df_clean['Attrition'] = df_clean['Attrition'].map({'Yes': 1, 'No': 0})
        
        # Step 3: Label Encoding for binary columns
        for col in self.binary_cols:
            if col in df_clean.columns:
                if col in self.label_encoders:
                    df_clean[col] = self.label_encoders[col].transform(df_clean[col])
                else:
                    raise ValueError(f"Label encoder for column '{col}' not found. Did you call fit()?")
        
        # Step 4: One-Hot Encoding for multi-class columns
        df_clean = pd.get_dummies(df_clean, columns=self.multi_class_cols, drop_first=True)
        
        # Step 5: Remove target if present
        if 'Attrition' in df_clean.columns:
            df_clean = df_clean.drop('Attrition', axis=1)
        
        # Step 6: Ensure all expected features are present
        for col in self.expected_features:
            if col not in df_clean.columns:
                df_clean[col] = 0
        
        # Keep only expected features in the correct order
        df_clean = df_clean[self.expected_features]
        
        # Step 7: Scale features
        X_scaled = self.scaler.transform(df_clean)
        
        return X_scaled

# ============================================================================
# LOAD MODEL AND CREATE PREPROCESSOR
# ============================================================================

model = None
preprocessor = None

def load_model_and_preprocessor():
    """Lazy load model and preprocessor"""
    global model, preprocessor
    
    if model is None or preprocessor is None:
        import warnings
        
        logger.info("Loading model and preprocessor...")
        
        # Check if files exist
        if not os.path.exists("best_attrition_model_gradient_boosting.pkl"):
            raise FileNotFoundError("Model file 'best_attrition_model_gradient_boosting.pkl' not found")
        
        # Load model
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            
            try:
                logger.info("  Loading model...")
                model = joblib.load("best_attrition_model_gradient_boosting.pkl")
                logger.info(f"  ✓ Model loaded: {type(model).__name__}")
            except ValueError as e:
                if "BitGenerator" in str(e):
                    logger.warning("  NumPy version mismatch. Attempting workaround...")
                    import numpy.random as npr
                    if not hasattr(npr, 'MT19937'):
                        npr.MT19937 = npr._mt19937.MT19937
                    model = joblib.load("best_attrition_model_gradient_boosting.pkl")
                    logger.info(f"  ✓ Model loaded with workaround")
                else:
                    raise
        
        # Load preprocessor from pickle OR create from scratch
        try:
            logger.info("  Loading preprocessor from pickle...")
            preprocessor = joblib.load("attrition_preprocessor.pkl")
            logger.info(f"  ✓ Preprocessor loaded from pickle: {len(preprocessor.expected_features)} features")
        except Exception as e:
            logger.warning(f"  Could not load preprocessor from pickle: {e}")
            logger.info("  Creating preprocessor from fitted scaler/encoders...")
            
            # Try to load the separate components if they exist
            preprocessor = AttritionPreprocessor()
            
            # Load the scaler and encoders that were saved separately
            if os.path.exists("preprocessor_scaler.pkl") and os.path.exists("preprocessor_encoders.pkl"):
                preprocessor.scaler = joblib.load("preprocessor_scaler.pkl")
                preprocessor.label_encoders = joblib.load("preprocessor_encoders.pkl")
                preprocessor.expected_features = joblib.load("preprocessor_features.pkl")
                logger.info(f"  ✓ Preprocessor created from components: {len(preprocessor.expected_features)} features")
            else:
                raise RuntimeError(
                    "Cannot create preprocessor. Please run the setup script to generate preprocessor files."
                )
        
        logger.info("✓ Model and preprocessor ready")
    
    return model, preprocessor

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class EmployeeDataRaw(BaseModel):
    """Raw employee data as it comes from the database"""
    Age: int = Field(..., ge=18, le=100, description="Employee age")
    BusinessTravel: str = Field(..., description="Travel frequency")
    DailyRate: int = Field(..., ge=0)
    Department: str = Field(..., description="Department")
    DistanceFromHome: int = Field(..., ge=0)
    Education: int = Field(..., ge=1, le=5)
    EducationField: str = Field(..., description="Field of education")
    EmployeeNumber: int = Field(..., ge=0)
    EnvironmentSatisfaction: int = Field(..., ge=1, le=4)
    Gender: str = Field(..., description="'Male' or 'Female'")
    HourlyRate: int = Field(..., ge=0)
    JobInvolvement: int = Field(..., ge=1, le=4)
    JobLevel: int = Field(..., ge=1, le=5)
    JobRole: str = Field(..., description="Job role title")
    JobSatisfaction: int = Field(..., ge=1, le=4)
    MaritalStatus: str = Field(..., description="'Single', 'Married', or 'Divorced'")
    MonthlyIncome: int = Field(..., ge=0)
    MonthlyRate: int = Field(..., ge=0)
    NumCompaniesWorked: int = Field(..., ge=0)
    OverTime: str = Field(..., description="'Yes' or 'No'")
    PercentSalaryHike: int = Field(..., ge=0)
    PerformanceRating: int = Field(..., ge=1, le=4)
    RelationshipSatisfaction: int = Field(..., ge=1, le=4)
    StockOptionLevel: int = Field(..., ge=0, le=3)
    TotalWorkingYears: int = Field(..., ge=0)
    TrainingTimesLastYear: int = Field(..., ge=0)
    WorkLifeBalance: int = Field(..., ge=1, le=4)
    YearsAtCompany: int = Field(..., ge=0)
    YearsInCurrentRole: int = Field(..., ge=0)
    YearsSinceLastPromotion: int = Field(..., ge=0)
    YearsWithCurrManager: int = Field(..., ge=0)
    
    # Optional fields
    EmployeeCount: Optional[int] = 1
    StandardHours: Optional[int] = 80
    Over18: Optional[str] = "Y"

class PredictionResponse(BaseModel):
    """API response with prediction results"""
    employee_number: int
    attrition_prediction: int
    attrition_probability: float
    risk_level: str
    message: str

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
def root():
    return {
        "message": "Employee Attrition Prediction API v2.0",
        "status": "active",
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    try:
        m, p = load_model_and_preprocessor()
        return {
            "status": "healthy",
            "model_loaded": m is not None,
            "preprocessor_loaded": p is not None,
            "expected_features": len(p.expected_features) if p else 0
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

# @app.post("/predict", response_model=PredictionResponse)
# def predict(employee_data: EmployeeDataRaw):
#     """Predict employee attrition from raw employee data."""
#     try:
#         # Load model and preprocessor
#         model, preprocessor = load_model_and_preprocessor()
        
#         # Convert Pydantic model to dict
#         raw_data = employee_data.model_dump()
        
#         logger.info(f"Received prediction request for Employee #{raw_data['EmployeeNumber']}")
        
#         # Preprocess the raw data
#         X_preprocessed = preprocessor.transform(raw_data)
#         logger.info(f"Preprocessing successful. Feature shape: {X_preprocessed.shape}")
        
#         # Make prediction
#         prediction = model.predict(X_preprocessed)[0]
#         probability = model.predict_proba(X_preprocessed)[0][1]
        
#         # Determine risk level
#         if probability < 0.3:
#             risk_level = "Low"
#         elif probability < 0.6:
#             risk_level = "Medium"
#         else:
#             risk_level = "High"
        
#         # Generate message
#         if prediction == 1:
#             message = f"Employee is predicted to leave. Risk level: {risk_level}"
#         else:
#             message = f"Employee is predicted to stay. Risk level: {risk_level}"
        
#         logger.info(f"Prediction: {prediction}, Probability: {probability:.4f}, Risk: {risk_level}")
        
#         return PredictionResponse(
#             employee_number=raw_data['EmployeeNumber'],
#             attrition_prediction=int(prediction),
#             attrition_probability=round(float(probability), 4),
#             risk_level=risk_level,
#             message=message
#         )
        
#     except Exception as e:
#         logger.error(f"Prediction error: {str(e)}", exc_info=True)
#         raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
def predict(employee_data: EmployeeDataRaw):
    """Predict employee attrition from raw employee data and provide explainability message."""
    try:
        # ============================================================
        # 1️⃣ Load model and preprocessor
        # ============================================================
        model, preprocessor = load_model_and_preprocessor()

        # Convert Pydantic object to dictionary
        raw_data = employee_data.model_dump()
        emp_id = raw_data.get("EmployeeNumber", -1)
        logger.info(f"Received prediction request for Employee #{emp_id}")

        # ============================================================
        # 2️⃣ Preprocess input
        # ============================================================
        X_preprocessed = preprocessor.transform(raw_data)
        X_df = pd.DataFrame(X_preprocessed, columns=preprocessor.expected_features)
        logger.info(f"Preprocessing successful. Shape: {X_df.shape}")

        # ============================================================
        # 3️⃣ Model prediction
        # ============================================================
        prediction = model.predict(X_preprocessed)[0]
        probability = float(model.predict_proba(X_preprocessed)[0][1])

        # Risk categorization
        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.6:
            risk_level = "Medium"
        else:
            risk_level = "High"

        # ============================================================
        # 4️⃣ SHAP EXPLANATION (Safe version)
        # ============================================================
        reason_text = "Model explanation unavailable."  # Default fallback
        
        try:
            import shap
            
            logger.info("Creating SHAP explainer...")
            # Use TreeExplainer for tree-based models (more stable)
            explainer = shap.TreeExplainer(model)
            logger.info("Computing SHAP values...")
            shap_values = explainer.shap_values(X_df)
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                # Binary classification returns [shap_class_0, shap_class_1]
                shap_vals = shap_values[1][0]  # Get class 1 (attrition) for first sample
            else:
                shap_vals = shap_values[0]
            
            logger.info(f"SHAP values computed successfully. Shape: {shap_vals.shape}")

            # Build feature impact table
            feature_impact = pd.DataFrame({
                "Feature": preprocessor.expected_features,
                "SHAP Value": shap_vals
            })
            feature_impact["abs_val"] = feature_impact["SHAP Value"].abs()

            # SMART SELECTION: Pick features relevant to the prediction
            if prediction == 1:
                # Employee will LEAVE → Show features INCREASING attrition (positive SHAP)
                top_features = feature_impact[feature_impact["SHAP Value"] > 0].sort_values(
                    "SHAP Value", ascending=False
                ).head(3)
            else:
                # Employee will STAY → Show features DECREASING attrition (negative SHAP)
                top_features = feature_impact[feature_impact["SHAP Value"] < 0].sort_values(
                    "SHAP Value", ascending=True
                ).head(3)

            # Fallback: if no features in expected direction, pick top absolute ones
            if top_features.empty:
                top_features = feature_impact.sort_values("abs_val", ascending=False).head(3)

            # Build human-readable explanation
            reason_phrases = []
            for _, row in top_features.iterrows():
                feat_name = row['Feature']
                shap_val = row['SHAP Value']
                
                # Make feature names more readable
                feat_display = feat_name.replace('_', ' ')
                
                if shap_val > 0:
                    reason_phrases.append(f"{feat_display} (pushes toward leaving)")
                else:
                    reason_phrases.append(f"{feat_display} (supports staying)")

            if reason_phrases:
                reason_text = "; ".join(reason_phrases)
            else:
                reason_text = "No significant factors identified."

        except Exception as shap_error:
            logger.error(f"SHAP explanation failed: {shap_error}", exc_info=True)
            reason_text = f"Model explanation unavailable: {str(shap_error)}"

        # ============================================================
        # 5️⃣ Construct response message
        # ============================================================
        if prediction == 1:
            message = (
                f"Employee is predicted to leave. "
                f"Risk level: {risk_level}. Top reasons: {reason_text}"
            )
        else:
            message = (
                f"Employee is predicted to stay. "
                f"Risk level: {risk_level}. Top reasons: {reason_text}"
            )

        # ============================================================
        # ✅ Return final structured response
        # ============================================================
        return PredictionResponse(
            employee_number=int(emp_id),
            attrition_prediction=int(prediction),
            attrition_probability=round(probability, 4),
            risk_level=risk_level,
            message=message
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/feature-info")
def feature_info():
    """Get information about expected features"""
    try:
        _, preprocessor = load_model_and_preprocessor()
        
        return {
            "total_features_after_preprocessing": len(preprocessor.expected_features),
            "preprocessing_steps": [
                "1. Drop columns: EmployeeCount, StandardHours, Over18, EmployeeNumber",
                "2. Label encode binary columns: Gender, OverTime",
                "3. One-hot encode: BusinessTravel, Department, EducationField, JobRole, MaritalStatus",
                "4. Standard scaling of all features"
            ],
            "binary_encoded_columns": preprocessor.binary_cols,
            "one_hot_encoded_columns": preprocessor.multi_class_cols,
            "sample_features": preprocessor.expected_features[:10]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load preprocessor: {str(e)}")
    
# ============================================================================
# Model input:

'''
    {
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
    }
'''
# ============================================================================

{
  "employee_number": 1,
  "attrition_prediction": 1,
  "attrition_probability": 0.7346,
  "risk_level": "High",
  "message": "Employee is predicted to leave. Risk level: High. Top reasons: Age (decreased attrition risk); DailyRate (decreased attrition risk); DistanceFromHome (decreased attrition risk)."
}