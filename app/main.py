import streamlit as st
import pandas as pd
import joblib
import os
from logic import preprocess_data # We still use this for cleaning the test set

# --- Load Saved Artifacts ---
MODEL_DIR = 'model'
try:
    model = joblib.load(os.path.join(MODEL_DIR, 'catboost_model.pkl'))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    X_columns = joblib.load(os.path.join(MODEL_DIR, 'X_columns.pkl'))
    artifacts_loaded = True
except Exception as e:
    artifacts_loaded = False
    st.error(f"Error loading model artifacts: {e}")

# --- Streamlit App UI ---
st.set_page_config(page_title="Credit Score Batch Prediction", layout="wide")
st.title('💳 Credit Score Batch Prediction App')

if not artifacts_loaded:
    st.error("Model artifacts not found! Please run `train_model.py` first to generate them.")
else:
    st.success("✅ Model loaded successfully! Upload your test data to get predictions.")

    uploaded_file = st.file_uploader("Choose a CSV file for prediction", type="csv")

    if uploaded_file is not None:
        # Load the uploaded data
        test_df = pd.read_csv(uploaded_file, low_memory=False)
        st.write("Uploaded Data Preview:")
        st.dataframe(test_df.head())

        # --- Preprocess the Test Data ---
        # We use the general pipeline from logic.py to clean the file
        processed_test_df = preprocess_data(test_df.copy())
        
        # Ensure all columns the model expects are present and in the correct order
        # This handles any columns that might be in test but not train, or vice-versa
        final_test_df = pd.DataFrame(columns=X_columns)
        for col in X_columns:
            if col in processed_test_df.columns:
                final_test_df[col] = processed_test_df[col]
            else:
                final_test_df[col] = 0 # Or np.nan, depending on imputation strategy
        
        # Handle any remaining NaNs after alignment
        final_test_df.fillna(0, inplace=True)

        # --- Make Predictions ---
        predictions_encoded = model.predict(final_test_df)
        predictions = label_encoder.inverse_transform(predictions_encoded)

        # --- Display Results ---
        results_df = test_df.copy()
        results_df['Predicted_Credit_Score'] = predictions

        st.write("---")
        st.header("Prediction Results")
        st.dataframe(results_df)
        st.write("---")
        st.header("Preprocessed Data")
        st.dataframe(final_test_df)

        # --- Provide Download Button ---
        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df_to_csv(results_df)

        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name='prediction_results.csv',
            mime='text/csv',
        )