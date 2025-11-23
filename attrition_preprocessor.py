import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

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