import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

def train_and_save_preprocessors(train_df):
    """
    Train and save the preprocessors (scaler and label encoder)
    This should be called during model training
    """
    # Create preprocessors directory
    os.makedirs("model", exist_ok=True)
    
    # Train and save label encoder for Gender
    label_encoder = LabelEncoder()
    label_encoder.fit(['Female', 'Male'])  # Ensure consistent encoding
    joblib.dump(label_encoder, "model/label_encoder.pkl")
    
    # Train and save scaler
    # Note: We need to fit the scaler on the original scale data
    # For deployment, we'll need to save the scaler fitted on unscaled data
    scaler = StandardScaler()
    
    # These are the features that will be scaled
    # We'll fit on dummy data with expected ranges for production
    dummy_data = pd.DataFrame({
        'CreditScore': np.random.randint(300, 850, 1000),
        'Age': np.random.randint(18, 100, 1000),
        'Balance': np.random.uniform(0, 250000, 1000),
        'CreditScore_Product': np.random.randint(300, 3400, 1000),  # CreditScore * NumOfProducts (1-4)
    })
    
    scaler.fit(dummy_data)
    joblib.dump(scaler, "model/scaler.pkl")
    
    return label_encoder, scaler
def preprocess_input(data):
    """
    Preprocess input data for prediction
    Input: dictionary with raw features
    Output: preprocessed dataframe ready for model prediction
    """
    # Load preprocessors
    model_path = "/app/model" if os.path.exists("/app/model") else "model"
    
    label_encoder = joblib.load(f"{model_path}/label_encoder.pkl")
    scaler = joblib.load(f"{model_path}/scaler.pkl")
    
    # Create dataframe from input
    df = pd.DataFrame([data])
    
    # Drop features not used by model (if they exist)
    features_to_drop = ['HasCrCard', 'Tenure', 'EstimatedSalary', 'RowNumber', 'CustomerId', 'Surname']
    df = df.drop(columns=[col for col in features_to_drop if col in df.columns])
    
    # Encode Gender
    if 'Gender' in df.columns:
        df['Gender'] = label_encoder.transform(df['Gender'])
    
    # One-hot encode Geography
    if 'Geography' in df.columns:
        df = pd.get_dummies(df, columns=['Geography'], drop_first=True)
        # Ensure we have the expected columns
        if 'Geography_Germany' not in df.columns:
            df['Geography_Germany'] = 0
        if 'Geography_Spain' not in df.columns:
            df['Geography_Spain'] = 0
    
    # Create engineered features
    df['CreditScore_Product'] = df['CreditScore'] * df['NumOfProducts']
    df['Gender_Activeness'] = df['Gender'] * df['IsActiveMember']
    
    # Scale continuous features
    features_to_scale = ['CreditScore', 'Age', 'Balance', 'CreditScore_Product']
    df[features_to_scale] = scaler.transform(df[features_to_scale])
    
    # Ensure column order matches training data
    expected_columns = [
        'CreditScore', 'Gender', 'Age', 'Balance', 'NumOfProducts',
        'IsActiveMember', 'Geography_Germany', 'Geography_Spain',
        'CreditScore_Product', 'Gender_Activeness'
    ]
    
    # Reorder columns to match training data
    df = df[expected_columns]
    
    return df