# FLASK_GCP_LAB - Customer Churn Prediction System

## Project Overview

This project implements a production-ready customer churn prediction system using Flask, Google Cloud Run, and machine learning. The system predicts whether a bank customer is likely to leave (churn) based on their account characteristics and behavior patterns.

### Key Features
- **Machine Learning Model**: Random Forest classifier with 85%+ accuracy
- **RESTful API**: Flask-based API for real-time predictions
- **Cloud Deployment**: Containerized deployment on Google Cloud Run
- **Preprocessing Pipeline**: Automated feature engineering and scaling
- **User Interface**: Interactive Streamlit web application
- **Auto-scaling**: Handles traffic spikes automatically
- **Cost-effective**: Pay-per-use pricing model

## System Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Streamlit  │────▶│  Cloud Run   │────▶│ Preprocessing│────▶│  ML Model    │
│     UI      │     │   Flask API  │     │   Pipeline   │     │   (RF)       │
└─────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                            │
                    ┌───────▼────────┐
                    │  Google Cloud  │
                    │  Container     │
                    │   Registry     │
                    └────────────────┘
```

## Project Structure

```
FLASK_GCP_LAB/
├── data/
│   └── train_data_cleaned.csv    # Preprocessed training data
├── model/
│   ├── model.pkl                 # Trained Random Forest model
│   ├── scaler.pkl               # StandardScaler for feature scaling
│   ├── label_encoder.pkl        # LabelEncoder for categorical variables
│   └── feature_names.pkl        # List of feature names
├── src/
│   ├── main.py                  # Flask API application
│   ├── train.py                 # Model training script
│   ├── predict.py               # Prediction logic
│   ├── preprocess.py            # Data preprocessing utilities
│   └── test_api.py              # API testing script
├── Dockerfile                    # Container configuration
├── requirements.txt              # Python dependencies
├── streamlit_app.py             # Streamlit frontend application
├── .gcloudignore                # Files to ignore during deployment
└── README.md                    # Project documentation
```

## How It Works

### 1. Data Flow
1. **Input**: Customer provides data through Streamlit UI
2. **API Request**: Data sent to Cloud Run endpoint
3. **Preprocessing**: Raw data is encoded, scaled, and engineered
4. **Prediction**: Model predicts churn probability
5. **Response**: Risk level and recommendations returned to UI

### 2. Feature Engineering
The system creates engineered features:
- `CreditScore_Product`: Credit Score × Number of Products
- `Gender_Activeness`: Gender × Active Member status
- One-hot encoding for Geography
- StandardScaling for numerical features

### 3. Risk Categorization
- **Low Risk**: < 30% churn probability 
- **Medium Risk**: 30-70% churn probability 
- **High Risk**: > 70% churn probability 

## Installation & Setup

### Prerequisites
- Python 3.9+
- Google Cloud Platform account with billing enabled
- Google Cloud SDK (gcloud CLI)
- Git

### Local Development Setup

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd FLASK_GCP_LAB
```

2. **Create virtual environment**
```bash
python -m venv mlops_lab1
source mlops_lab1/bin/activate  # On Windows: mlops_lab1\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Train the model**
```bash
python src/train.py
```

5. **Test locally**
```bash
# Start Flask API
python src/main.py

# In another terminal, test the API
python src/test_api.py

# Run Streamlit UI
streamlit run streamlit_app.py
```

## Google Cloud Deployment

### 1. Initial GCP Setup

```bash
# Authenticate with GCP
gcloud auth login
gcloud auth application-default login

# Set project
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com
```

### 2. Build and Deploy

```bash
# Build container image
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/customer-churn-app

# Deploy to Cloud Run
gcloud run deploy customer-churn-app \
    --image gcr.io/YOUR_PROJECT_ID/customer-churn-app \
    --region us-east1 \
    --allow-unauthenticated \
    --port 8501 \
    --memory 1Gi

# Get service URL
gcloud run services describe customer-churn-app \
    --region us-east1 \
    --format 'value(status.url)'
```

### 3. Update Streamlit with Deployed URL

Edit `streamlit_app.py` and update the API_URL:
```python
API_URL = 'https://your-service-url.run.app/predict'
```

## API Documentation

### Base URL
```
https://customer-churn-app-h52fnzgp4a-ue.a.run.app
```

### Endpoints

#### 1. Health Check
- **GET** `/health`
- **Response**: `{"status": "healthy"}`

#### 2. Get API Info
- **GET** `/`
- **Response**: API documentation and expected inputs

#### 3. Predict Churn
- **POST** `/predict`
- **Content-Type**: `application/json`
- **Request Body**:
```json
{
    "CreditScore": 600,
    "Geography": "Germany",
    "Gender": "Female",
    "Age": 40,
    "Balance": 75000,
    "NumOfProducts": 2,
    "IsActiveMember": 1
}
```
- **Response**:
```json
{
    "will_churn": "No",
    "churn_risk": "Low",
    "churn_probability": "11.50%",
    "retention_probability": "88.50%",
    "raw_prediction": 0
}
```

## Testing

### API Testing with cURL
```bash
curl -X POST https://customer-churn-app-h52fnzgp4a-ue.a.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "CreditScore": 600,
    "Geography": "Germany",
    "Gender": "Female",
    "Age": 40,
    "Balance": 75000,
    "NumOfProducts": 2,
    "IsActiveMember": 1
  }'
```

### Python Testing
```python
import requests

response = requests.post(
    'https://customer-churn-app-h52fnzgp4a-ue.a.run.app/predict',
    json={
        "CreditScore": 600,
        "Geography": "Germany",
        "Gender": "Female",
        "Age": 40,
        "Balance": 75000,
        "NumOfProducts": 2,
        "IsActiveMember": 1
    }
)
print(response.json())
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Billing Not Enabled
```
ERROR: Billing account for project is not found
```
**Solution**: Enable billing in GCP Console → Billing → Link billing account

#### 2. Port Configuration Issues
```
Container failed to start and listen on port
```
**Solution**: Ensure Flask app uses `PORT` environment variable:
```python
app.run(port=int(os.environ.get("PORT", 8501)))
```

#### 3. Module Import Errors
```
ModuleNotFoundError: No module named 'preprocess'
```
**Solution**: Add proper path configuration:
```python
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
```

#### 4. Model File Not Found
```
FileNotFoundError: model/model.pkl
```
**Solution**: Run `python src/train.py` to generate model files

## Cost Optimization

- **Cloud Run**: First 2 million requests/month free
- **Set max instances**: `gcloud run services update customer-churn-app --max-instances=10`
- **Enable CPU throttling**: Reduces costs during idle time
- **Use minimum instances = 0**: Scale to zero when not in use

## Performance Metrics

- **Model Accuracy**: ~85%
- **API Response Time**: < 200ms average
- **Container Size**: ~500MB
- **Memory Usage**: < 512MB
- **Cold Start Time**: ~5 seconds

## Security Considerations

- API is currently unauthenticated (public access)
- For production, consider:
  - Adding API authentication (JWT/OAuth)
  - Implementing rate limiting
  - Using Cloud Armor for DDoS protection
  - Enabling HTTPS only
  - Setting up monitoring and alerts

## Dependencies

### Main Libraries
- **Flask**: Web framework for API
- **scikit-learn**: Machine learning model
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **joblib**: Model serialization
- **streamlit**: User interface
- **requests**: HTTP client

### Full Requirements
See `requirements.txt` for complete list with versions.

## CI/CD Pipeline (Optional Enhancement)

To set up automated deployment:

1. **Create Cloud Build trigger**:
```yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/customer-churn-app', '.']
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/customer-churn-app']
  - name: 'gcr.io/cloud-builders/gcloud'
    args: ['run', 'deploy', 'customer-churn-app', '--image', 'gcr.io/$PROJECT_ID/customer-churn-app', '--region', 'us-east1']
```

2. **Connect to GitHub repository**
3. **Set trigger on push to main branch**

## Future Enhancements

- [ ] Add model versioning and A/B testing
- [ ] Implement batch prediction endpoint
- [ ] Add database for prediction logging
- [ ] Create model retraining pipeline
- [ ] Add explainability features (SHAP/LIME)
- [ ] Implement real-time monitoring dashboard
- [ ] Add customer segmentation features
- [ ] Deploy Streamlit to Cloud Run
- [ ] Add email alerts for high-risk customers
- [ ] Implement feedback loop for model improvement

## License

This project is for educational purposes as part of IE7374 MLOps course.

## Contributors

- Dennis M Jose (@dennismjose)

## Support

For issues or questions:
1. Check the Troubleshooting section
2. Review API documentation
3. Check Cloud Run logs: `gcloud logging read`
4. Create an issue in the repository

---

**Last Updated**: September 2025
**Version**: 1.0.0
**Status**: Production Ready 