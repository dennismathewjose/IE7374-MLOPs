from flask import Flask, request, jsonify
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from predict import predict_churn

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Customer Churn Prediction API',
        'endpoints': {
            '/': 'This help message',
            '/predict': 'POST endpoint for churn prediction',
            '/health': 'Health check endpoint'
        },
        'expected_input': {
            'CreditScore': 'int (300-850)',
            'Geography': 'string (France/Germany/Spain)',
            'Gender': 'string (Male/Female)',
            'Age': 'int (18-100)',
            'Balance': 'float',
            'NumOfProducts': 'int (1-4)',
            'IsActiveMember': 'int (0 or 1)'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['CreditScore', 'Geography', 'Gender', 'Age', 
                          'Balance', 'NumOfProducts', 'IsActiveMember']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Make prediction with raw data
        result = predict_churn(data)
        
        # Format response
        response = {
            'will_churn': 'Yes' if result['prediction'] == 1 else 'No',
            'churn_risk': 'High' if result['churn_probability'] > 0.7 else 
                         'Medium' if result['churn_probability'] > 0.3 else 'Low',
            'churn_probability': f"{result['churn_probability']:.2%}",
            'retention_probability': f"{result['retention_probability']:.2%}",
            'raw_prediction': result['prediction']
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8501)))