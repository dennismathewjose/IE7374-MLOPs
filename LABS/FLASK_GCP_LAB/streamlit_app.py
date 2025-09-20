import streamlit as st
import requests
import os

st.set_page_config(page_title="Customer Churn Predictor", page_icon="🏦")

st.title('Customer Churn Prediction')
st.markdown("### Predict if a customer is likely to leave the bank")

# Create two columns for input fields
col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Information")
    credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=650, step=10,
                                   help="Customer's credit score (300-850)")
    age = st.number_input('Age', min_value=18, max_value=100, value=35, step=1,
                         help="Customer's age in years")
    balance = st.number_input('Account Balance ($)', min_value=0.0, max_value=500000.0, value=75000.0, step=1000.0,
                             help="Current account balance")
    num_products = st.selectbox('Number of Products', options=[1, 2, 3, 4], index=1,
                                help="Number of bank products the customer uses")

with col2:
    st.subheader("Demographics")
    gender = st.selectbox('Gender', options=['Female', 'Male'],
                         help="Customer's gender")
    
    geography = st.selectbox('Country', options=['France', 'Germany', 'Spain'],
                           help="Customer's country of residence")
    
    st.subheader("Account Status")
    is_active = st.checkbox('Active Member', value=True,
                           help="Is the customer actively using bank services?")

# Prediction button
if st.button('Predict Churn Risk', type='primary', use_container_width=True):
    # Prepare data for API
    data = {
        'CreditScore': credit_score,
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Balance': balance,
        'NumOfProducts': num_products,
        'IsActiveMember': 1 if is_active else 0
    }
    
    try:
        # For local testing use:
        # API_URL = 'http://localhost:8501/predict'
        
        # For deployed version (UPDATE THIS WITH YOUR ACTUAL URL):
        API_URL = 'https://customer-churn-app-h52fnzgp4a-ue.a.run.app/predict'
        
        with st.spinner('Analyzing customer data...'):
            response = requests.post(API_URL, json=data, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            
            # Display results
            st.markdown("---")
            
            # Main prediction with color coding
            if result['raw_prediction'] == 1:
                st.error(f" **Churn Prediction: {result['will_churn']}**")
                risk_color = "" if result['churn_risk'] == 'High' else ""
            else:
                st.success(f"**Churn Prediction: {result['will_churn']}**")
                risk_color = ""
            
            st.markdown(f"### {risk_color} Risk Level: **{result['churn_risk']}**")
            
            # Probability metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Churn Probability", result['churn_probability'])
            with col2:
                st.metric("Retention Probability", result['retention_probability'])
            
            # Recommendations
            if result['raw_prediction'] == 1:
                st.markdown("### Recommended Actions:")
                if result['churn_risk'] == 'High':
                    st.warning("""
                    **Immediate Action Required:**
                    - Contact customer within 24 hours
                    - Offer retention incentives
                    - Assign dedicated relationship manager
                    """)
                else:
                    st.info("""
                    **Preventive Measures:**
                    - Schedule follow-up call
                    - Offer loyalty rewards
                    - Review product portfolio
                    """)
                
        else:
            st.error(f'Error: {response.status_code}')
            if response.text:
                st.error(f'Details: {response.text}')
            
    except requests.exceptions.Timeout:
        st.error('Request timed out. Please try again.')
    except requests.exceptions.ConnectionError:
        st.error('Cannot connect to the prediction service. Please check if the API is running.')
    except Exception as e:
        st.error(f'Unexpected error: {str(e)}')

# Sidebar with information
with st.sidebar:
    st.markdown("##About This Model")
    st.markdown("""
    **Algorithm:** Random Forest
    
    **Risk Categories:**
    - **Low**: < 30% churn probability
    - **Medium**: 30-70% probability  
    - **High**: > 70% probability
    
    **Features Used:**
    - Credit Score
    - Age & Gender
    - Account Balance
    - Number of Products
    - Activity Status
    - Geographic Location
    """)