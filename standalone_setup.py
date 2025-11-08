"""
Standalone setup script to create preprocessor components from scratch.
This script downloads the dataset, fits the preprocessor, and saves components.

Run this ONCE: python standalone_setup.py
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ============================================================================
# PREPROCESSOR CLASS (EMBEDDED)
# ============================================================================

class AttritionPreprocessor:
    """Preprocessing pipeline for IBM HR Attrition dataset."""
    
    def __init__(self):
        self.cols_to_drop = ['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber']
        self.binary_cols = ['Gender', 'OverTime']
        self.multi_class_cols = ['BusinessTravel', 'Department', 'EducationField', 
                                  'JobRole', 'MaritalStatus']
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.expected_features = None
        
    def fit(self, df: pd.DataFrame):
        """Fit the preprocessor on training data."""
        df_clean = df.copy()
        
        # Drop columns
        df_clean = df_clean.drop(columns=self.cols_to_drop, errors='ignore')
        
        # Encode target
        if 'Attrition' in df_clean.columns:
            df_clean['Attrition'] = df_clean['Attrition'].map({'Yes': 1, 'No': 0})
        
        # Label encoding for binary columns
        for col in self.binary_cols:
            if col in df_clean.columns:
                le = LabelEncoder()
                df_clean[col] = le.fit_transform(df_clean[col])
                self.label_encoders[col] = le
        
        # One-hot encoding for multi-class columns
        df_clean = pd.get_dummies(df_clean, columns=self.multi_class_cols, drop_first=True)
        
        # Separate features
        if 'Attrition' in df_clean.columns:
            X = df_clean.drop('Attrition', axis=1)
        else:
            X = df_clean
        
        # Store expected features
        self.expected_features = X.columns.tolist()
        
        # Fit scaler
        self.scaler.fit(X)
        
        return self
    
    def transform(self, data):
        """Transform raw employee data to preprocessed features."""
        from typing import Dict, Any
        
        # Convert dict to DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()
        
        # Drop columns
        df_clean = df.drop(columns=self.cols_to_drop, errors='ignore')
        
        # Encode target if present
        if 'Attrition' in df_clean.columns:
            df_clean['Attrition'] = df_clean['Attrition'].map({'Yes': 1, 'No': 0})
        
        # Label encoding for binary columns
        for col in self.binary_cols:
            if col in df_clean.columns:
                if col in self.label_encoders:
                    df_clean[col] = self.label_encoders[col].transform(df_clean[col])
                else:
                    raise ValueError(f"Label encoder for column '{col}' not found")
        
        # One-hot encoding for multi-class columns
        df_clean = pd.get_dummies(df_clean, columns=self.multi_class_cols, drop_first=True)
        
        # Remove target if present
        if 'Attrition' in df_clean.columns:
            df_clean = df_clean.drop('Attrition', axis=1)
        
        # Ensure all expected features are present
        for col in self.expected_features:
            if col not in df_clean.columns:
                df_clean[col] = 0
        
        # Keep only expected features in correct order
        df_clean = df_clean[self.expected_features]
        
        # Scale features
        X_scaled = self.scaler.transform(df_clean)
        
        return X_scaled

# ============================================================================
# MAIN SETUP FUNCTION
# ============================================================================

def setup_preprocessor():
    """Download data, fit preprocessor, save components."""
    
    print("="*80)
    print("ATTRITION API - PREPROCESSOR SETUP")
    print("="*80)
    
    # Step 1: Download dataset
    print("\n[1/5] Downloading dataset from Kaggle...")
    try:
        import kagglehub
        path = kagglehub.dataset_download("pavansubhasht/ibm-hr-analytics-attrition-dataset")
        print(f"      ✓ Dataset downloaded to: {path}")
    except ImportError:
        print("      ✗ kagglehub not installed")
        print("\n      Installing kagglehub...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'kagglehub'])
        import kagglehub
        path = kagglehub.dataset_download("pavansubhasht/ibm-hr-analytics-attrition-dataset")
        print(f"      ✓ Dataset downloaded to: {path}")
    except Exception as e:
        print(f"      ✗ Error: {e}")
        return False
    
    # Step 2: Load dataset
    print("\n[2/5] Loading CSV file...")
    try:
        csv_files = list(Path(path).glob('*.csv'))
        if not csv_files:
            print("      ✗ No CSV files found!")
            return False
        
        df = pd.read_csv(csv_files[0])
        print(f"      ✓ Loaded: {csv_files[0].name}")
        print(f"      ✓ Shape: {df.shape}")
        print(f"      ✓ Columns: {len(df.columns)}")
    except Exception as e:
        print(f"      ✗ Error loading CSV: {e}")
        return False
    
    # Step 3: Fit preprocessor
    print("\n[3/5] Fitting preprocessor...")
    try:
        preprocessor = AttritionPreprocessor()
        preprocessor.fit(df)
        print(f"      ✓ Preprocessor fitted")
        print(f"      ✓ Expected features after preprocessing: {len(preprocessor.expected_features)}")
        print(f"      ✓ Binary columns: {preprocessor.binary_cols}")
        print(f"      ✓ Multi-class columns: {preprocessor.multi_class_cols}")
    except Exception as e:
        print(f"      ✗ Error fitting: {e}")
        return False
    
    # Step 4: Save components
    print("\n[4/5] Saving preprocessor components...")
    try:
        joblib.dump(preprocessor.scaler, "preprocessor_scaler.pkl")
        print("      ✓ Saved: preprocessor_scaler.pkl")
        
        joblib.dump(preprocessor.label_encoders, "preprocessor_encoders.pkl")
        print("      ✓ Saved: preprocessor_encoders.pkl")
        
        joblib.dump(preprocessor.expected_features, "preprocessor_features.pkl")
        print("      ✓ Saved: preprocessor_features.pkl")
        
        # Also save the full preprocessor for backup
        joblib.dump(preprocessor, "attrition_preprocessor_local.pkl")
        print("      ✓ Saved: attrition_preprocessor_local.pkl (backup)")
        
    except Exception as e:
        print(f"      ✗ Error saving: {e}")
        return False
    
    # Step 5: Verify
    print("\n[5/5] Verifying setup...")
    try:
        # Test loading
        scaler = joblib.load("preprocessor_scaler.pkl")
        encoders = joblib.load("preprocessor_encoders.pkl")
        features = joblib.load("preprocessor_features.pkl")
        
        print(f"      ✓ Scaler loaded: {type(scaler).__name__}")
        print(f"      ✓ Encoders loaded: {len(encoders)} encoders")
        print(f"      ✓ Features loaded: {len(features)} features")
        
        # Test transformation
        test_row = df.iloc[0].to_dict()
        
        # Create a test preprocessor and load components
        test_preprocessor = AttritionPreprocessor()
        test_preprocessor.scaler = scaler
        test_preprocessor.label_encoders = encoders
        test_preprocessor.expected_features = features
        
        X_test = test_preprocessor.transform(test_row)
        print(f"      ✓ Test transformation successful: {X_test.shape}")
        
    except Exception as e:
        print(f"      ✗ Verification failed: {e}")
        return False
    
    # Success message
    print("\n" + "="*80)
    print("✓ SETUP COMPLETE!")
    print("="*80)
    print("\nFiles created:")
    print("  1. preprocessor_scaler.pkl       - Standard scaler")
    print("  2. preprocessor_encoders.pkl     - Label encoders")
    print("  3. preprocessor_features.pkl     - Feature names")
    print("  4. attrition_preprocessor_local.pkl - Full backup")
    print("\nYour directory should also have:")
    print("  - best_attrition_model_gradient_boosting.pkl (your model)")
    print("  - main.py (FastAPI application)")
    print("\nNext step:")
    print("  Run: uvicorn main:app --reload")
    print("\nThe API is ready to accept raw employee data! 🎉")
    
    return True

# ============================================================================
# RUN SETUP
# ============================================================================

if __name__ == "__main__":
    import sys
    
    try:
        success = setup_preprocessor()
        if not success:
            print("\n❌ Setup failed. Please check the errors above.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)