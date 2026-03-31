import pandas as pd
import joblib
import os
from category_encoders import TargetEncoder
from imblearn.over_sampling import SMOTE

class DataProcessor:
    def __init__(self):
        self.encoder = TargetEncoder()
        self.smote = SMOTE(random_state=42)
        # We drop 'month' and 'day' to keep the model logical
        self.drop_cols = ['month', 'day']

    def clean_and_encode(self, df, is_train=True):
        # 1. Clean column names (removing quotes if present)
        df.columns = [c.strip('"') for c in df.columns]
        
        # 2. Drop the seasonal columns IMMEDIATELY
        # This reduces our feature set from 16 to 14
        df_cleaned = df.drop(columns=self.drop_cols, errors='ignore')
        
        if is_train:
            # 3. Training Mode Logic
            if 'y' not in df_cleaned.columns:
                raise ValueError("Target column 'y' not found in training data!")
                
            y = df_cleaned['y'].map({'yes': 1, 'no': 0})
            X = df_cleaned.drop(columns=['y'])
            
            # Perform Encoding
            X_encoded = self.encoder.fit_transform(X, y)
            
            # Handle class imbalance
            X_resampled, y_resampled = self.smote.fit_resample(X_encoded, y)
            
            # Save the fitted encoder for the UI to use
            os.makedirs('models', exist_ok=True)
            joblib.dump(self.encoder, 'models/encoder.pkl')
            
            return X_resampled, y_resampled
            
        else:
            # 4. UI/Test Mode Logic (Fixes the 15 vs 14 error)
            # Remove 'y' if it exists in the uploaded CSV so it doesn't count as a feature
            X_test = df_cleaned.drop(columns=['y'], errors='ignore')
            
            # Load the exact encoder used during training
            if not os.path.exists('models/encoder.pkl'):
                raise FileNotFoundError("Encoder not found! Run main.py first.")
                
            loaded_encoder = joblib.load('models/encoder.pkl')
            
            # Transform the data into the 14-feature format
            return loaded_encoder.transform(X_test)