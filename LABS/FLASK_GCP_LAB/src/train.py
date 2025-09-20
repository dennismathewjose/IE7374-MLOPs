import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

def run_training():
    # Load raw data
    raw_data = pd.read_csv("/Users/dennis_m_jose/Documents/GitHub/IE7374-MLOPs/LABS/FLASK_GCP_LAB/src/data/Churn_Modelling.csv")  # Assuming you have the raw data
    
    # Make a copy for processing
    train_df = raw_data.copy()
    
    # Drop unnecessary features
    features_to_drop = ['RowNumber', 'CustomerId', 'Surname', 'HasCrCard', 'Tenure', 'EstimatedSalary']
    train_df = train_df.drop(columns=[col for col in features_to_drop if col in train_df.columns])
    
    # Encode Gender variable using LabelEncoder
    label_encoder = LabelEncoder()
    train_df['Gender'] = label_encoder.fit_transform(train_df['Gender'])
    
    # One-hot encoding for Geography
    train_df = pd.get_dummies(train_df, columns=['Geography'], drop_first=True)
    
    # Create engineered features
    train_df['Age_Balance'] = train_df['Age'] * train_df['Balance']
    train_df['CreditScore_Product'] = train_df['CreditScore'] * train_df['NumOfProducts']
    train_df['Gender_Activeness'] = train_df['Gender'] * train_df['IsActiveMember']
    
    # Scale continuous features
    features_to_scale = ['CreditScore', 'Age', 'Balance', 'Age_Balance', 'CreditScore_Product']
    
    scaler = StandardScaler()
    train_df[features_to_scale] = scaler.fit_transform(train_df[features_to_scale])
    
    # Remove Age_Balance due to multicollinearity
    train_df.drop(columns=['Age_Balance'], inplace=True)
    print("Age_Balance feature removed due to high multicollinearity.")
    
    # Save cleaned data
    os.makedirs("data", exist_ok=True)
    train_df.to_csv('data/train_data_cleaned.csv', index=False)
    
    # Create model directory
    os.makedirs("model", exist_ok=True)
    
    # Save preprocessors for production use
    # We need to save the scaler fitted on UNSCALED data for production
    # So let's create new ones for production
    scaler_prod = StandardScaler()
    unscaled_features = raw_data[['CreditScore', 'Age', 'Balance']].copy()
    unscaled_features['CreditScore_Product'] = raw_data['CreditScore'] * raw_data['NumOfProducts']
    scaler_prod.fit(unscaled_features)
    joblib.dump(scaler_prod, "model/scaler.pkl")
    
    # Save label encoder
    joblib.dump(label_encoder, "model/label_encoder.pkl")
    
    # Split features and target
    X = train_df.drop("Exited", axis=1)
    y = train_df["Exited"]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the RandomForestClassifier
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the trained model
    joblib.dump(model, "model/model.pkl")
    print("Model trained and saved successfully!")
    
    # Print accuracy
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.4f}")
    
    # Save feature names for reference
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, "model/feature_names.pkl")
    print(f"Features used: {feature_names}")

if __name__ == "__main__":
    run_training()
