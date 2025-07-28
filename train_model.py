import pandas as pd
import warnings
import os
import joblib
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
from app.logic import preprocess_data # <--- KEY CHANGE: Import the unified preprocessing function

warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_FILE_PATH = 'data/train.csv'
MODEL_DIR = 'model'
TARGET_COLUMN = 'Credit_Score'

def train_on_full_data():
    """
    Trains the model on the entire training dataset using the unified
    preprocessing logic and saves the artifacts.
    """
    print("🚀 Starting training process on the full dataset...")

    # --- Load Data ---
    df = pd.read_csv(DATA_FILE_PATH, low_memory=False)
    print(f"Loaded {len(df)} rows from {DATA_FILE_PATH}")

    # --- UNIFIED PREPROCESSING ---
    # Use the same function as the Streamlit app to ensure consistency
    print("Applying unified preprocessing pipeline...")
    processed_df = preprocess_data(df.copy())

    # Drop Customer_ID as it's not a feature for the model
    if 'Customer_ID' in processed_df.columns:
        processed_df = processed_df.drop(columns=['Customer_ID'])

    # Final imputation for any remaining NaNs after processing
    for col in processed_df.select_dtypes(include=['number']).columns:
        processed_df[col].fillna(processed_df[col].median(), inplace=True)
    for col in processed_df.select_dtypes(include=['object']).columns:
        if col != TARGET_COLUMN:
            processed_df[col].fillna(processed_df[col].mode()[0], inplace=True)

    # --- Prepare Final Dataset ---
    X = processed_df.drop(TARGET_COLUMN, axis=1)
    y = processed_df[TARGET_COLUMN]

    # Encode target variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # --- Train CatBoost Model ---
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    cat_features_indices = [X.columns.get_loc(col) for col in categorical_cols]

    print("\n--- Training CatBoost Model on Full Data ---")
    model = CatBoostClassifier(
        task_type="GPU",
        iterations=4000,
        learning_rate=0.09,
        depth=7,
        cat_features=cat_features_indices,
        random_seed=42,
        eval_metric='MultiClass',
        verbose=100
    )
    model.fit(X, y_encoded)

    # --- Save Artifacts ---
    print("\n💾 Saving model and artifacts...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, 'catboost_model.pkl'))
    joblib.dump(le, os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    joblib.dump(X.columns.tolist(), os.path.join(MODEL_DIR, 'X_columns.pkl'))
    
    print(f"🎉 All artifacts saved in the '{MODEL_DIR}/' directory.")

if __name__ == '__main__':
    train_on_full_data()