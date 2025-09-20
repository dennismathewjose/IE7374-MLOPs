import numpy as np
import pandas as pd
import joblib
import os
import sys

# Add the src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess import preprocess_input

# Load the trained model
model_path = "/app/model" if os.path.exists("/app/model") else "model"
model = joblib.load(f"{model_path}/model.pkl")

def predict_churn(customer_data):
    """
    Predict customer churn based on raw input features
    Input: dictionary with raw customer data
    Returns: prediction results
    """
    # Preprocess the input data
    processed_data = preprocess_input(customer_data)
    
    # Make prediction
    prediction = model.predict(processed_data)
    probability = model.predict_proba(processed_data)
    
    return {
        'prediction': int(prediction[0]),
        'churn_probability': float(probability[0][1]),
        'retention_probability': float(probability[0][0])
    }