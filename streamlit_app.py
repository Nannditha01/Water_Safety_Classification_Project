import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved model and scaler
model = joblib.load('best_water_quality_model.pkl')
scaler = joblib.load('scaler.pkl')

# App Title
st.title("🚰 Water Safety Classification")
st.write("Predict whether a water sample is Safe or Unsafe.")

#File uploader
uploaded_file = st.file_uploader("Upload a CSV file containing water samples", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview:")
    st.dataframe(data.head())

    # Data cleaning
    data['ammonia'] = pd.to_numeric(data['ammonia'], errors='coerce')
    data.dropna(inplace=True)

    # Check columns in the uploaded data
    required_columns = scaler.feature_names_in_  # Columns expected by the scaler (same as used during fitting)
    missing_columns = set(required_columns) - set(data.columns)
    
    if missing_columns:
        st.error(f"Missing columns in the uploaded data: {', '.join(missing_columns)}")
    else:
        # Ensure columns are in the right order for transformation
        data = data[required_columns]  # Reorder columns to match the model's expected input

        # Scale features
        data_scaled = scaler.transform(data)
            # Predict
    predictions = model.predict(data_scaled)
    prediction_labels = np.where(predictions == 1, 'Safe', 'Unsafe')

    # Results
    data['Water_Safety_Prediction'] = prediction_labels
    st.subheader("Prediction Results:")
    st.dataframe(data)
    
    # Display results
    st.subheader("Prediction Results")
    
    if predictions[0] == 1:
        st.success("✅ The water is predicted to be SAFE for consumption")
    else:
        st.error("❌ The water is predicted to be UNSAFE for consumption")
    

    # Download results
    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions", csv, "water_safety_predictions.csv", "text/csv")

    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info("This app uses a machine learning model to classify water samples as Safe or Unsafe.\n\nDeveloped for the Water Safety Classification Project.")

