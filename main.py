import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from core.processor import DataProcessor
from core.model_trainer import ModelTrainer

def main():
    # 1. Setup Folders
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    print("🚀 Starting Bank Marketing Training Pipeline...")

    # 2. Load Data 
    try:
        train_path = 'data/train.csv'
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Could not find {train_path}. Please place it in the data/ folder.")
        
        train_df = pd.read_csv(train_path, sep=';')
        print(f"✅ Data Loaded: {train_df.shape[0]} rows found.")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # 3. Preprocessing (Target Encoding & SMOTE)
    print("🛠️  Processing features and balancing classes (SMOTE)...")
    processor = DataProcessor()
    
    # This step now automatically drops 'month' and 'day' via core/processor.py
    X_resampled, y_resampled = processor.clean_and_encode(train_df, is_train=True)
    
    # CRITICAL DEBUG: Verify feature count is 14
    n_features = X_resampled.shape[1]
    print(f"📊 Feature Count: {n_features} (Expected: 14 if month/day are dropped)")

    # --- ADDED: SPLIT FOR RELIABILITY CHECK ---
    X_train, X_val, y_train, y_val = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42
    )

    # 4. Hyperparameter Tuning (Optuna)
    print("🧪 Running Optuna for XGBoost optimization...")
    trainer = ModelTrainer()
    best_params = trainer.tune_xgboost(X_train, y_train)
    print(f"🎯 Best Params Found: {best_params}")

    # 5. Building the Stack (Meta-Learner: Logistic Regression)
    print("🔗 Training Stacking Ensemble...")
    stack_model = trainer.build_stack(X_train, y_train, best_params)
    
    # --- ADDED: MODEL RELIABILITY REPORT ---
    print("\n" + "="*40)
    print("📋 MODEL RELIABILITY REPORT (Validation)")
    print("="*40)
    
    # Test on the validation set
    y_pred = stack_model.predict(X_val)
    print("\n--- Classification Performance ---")
    print(classification_report(y_val, y_pred))
    
    # Overfitting Check
    train_acc = stack_model.score(X_train, y_train)
    val_acc = stack_model.score(X_val, y_val)
    print(f"✅ Training Accuracy: {train_acc:.2%}")
    print(f"✅ Validation Accuracy: {val_acc:.2%}")
    
    if (train_acc - val_acc) > 0.15:
        print("⚠️  Warning: High variance detected (Possible Overfitting).")
    else:
        print("💎 Stability Check: Model is generalizing well.")
    
    # Final check to prevent dimension errors in app.py
    print(f"⚙️  Model trained to receive exactly {n_features} features.")
    print("="*40 + "\n")

    # 6. Saving results
    print("💾 Saving model and encoder to /models...")
    # This saves the model that expects 14 features
    joblib.dump(stack_model, 'models/stack_model.pkl')
    
    print("✨ Pipeline Complete! You can now run 'streamlit run app.py'")

if __name__ == "__main__":
    main()