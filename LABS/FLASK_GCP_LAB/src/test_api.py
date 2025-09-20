import requests
import json

# Test customer data
test_customer = {
    'CreditScore': 600,
    'Geography': 'Germany',
    'Gender': 'Female',
    'Age': 40,
    'Balance': 75000.0,
    'NumOfProducts': 2,
    'IsActiveMember': 1
}

# Local test URL
url = 'http://127.0.0.1:8501/predict'

print("Testing Customer Churn Prediction API...")
print(f"Input data: {json.dumps(test_customer, indent=2)}")
print("-" * 50)

try:
    # Send POST request with JSON data
    response = requests.post(url, json=test_customer)
    
    if response.status_code == 200:
        result = response.json()
        print("API Test Successful!")
        print(f"Will Churn: {result['will_churn']}")
        print(f"Churn Risk: {result['churn_risk']}")
        print(f"Churn Probability: {result['churn_probability']}")
        print(f"Retention Probability: {result['retention_probability']}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"Connection Error: {str(e)}")
    print("Make sure the Flask server is running: python src/main.py")