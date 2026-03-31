import pandas as pd
import joblib
import os
from category_encoders import TargetEncoder
from imblearn.over_sampling import SMOTE

class DataProcessor:
    def __init__(self):
        self.encoder = TargetEncoder()
        self.smote = SMOTE(random_state=42)
        self.drop_cols = ['month', 'day']

    def clean_and_encode(self, df, is_train=True):
        df.columns = [c.strip('"') for c in df.columns]
        df_cleaned = df.drop(columns=self.drop_cols, errors='ignore')
        
        if is_train:
            if 'y' not in df_cleaned.columns:
                raise ValueError("Target column 'y' not found in training data!")
                
            y = df_cleaned['y'].map({'yes': 1, 'no': 0})
            X = df_cleaned.drop(columns=['y'])
            
            X_encoded = self.encoder.fit_transform(X, y)
            X_resampled, y_resampled = self.smote.fit_resample(X_encoded, y)
            
            os.makedirs('models', exist_ok=True)
            joblib.dump(self.encoder, 'models/encoder.pkl')
            return X_resampled, y_resampled
        else:
            X_test = df_cleaned.drop(columns=['y'], errors='ignore')
            if not os.path.exists('models/encoder.pkl'):
                raise FileNotFoundError("Encoder not found! Run main.py first.")
            loaded_encoder = joblib.load('models/encoder.pkl')
            return loaded_encoder.transform(X_test)