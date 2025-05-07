import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st


# load the model
model = tf.keras.models.load_model('model.h5')

# Load the encoders and Scaler

with open('label_endoder_gender.pkl', 'rb')as file:
    label_endoder_gender=pickle.load(file)
    
with open('onehot_encode_geo.pkl', 'rb')as file:
    onehot_encode_geo=pickle.load(file)
    
    
with open('scaler.pkl', 'rb')as file:
    scaler=pickle.load(file)   
    
# Streamlit app
st.title('Customer Churn Prediction')

# Create columns for better layout
col1, col2 = st.columns(2)

with col1:
    # User Inputs
    geography = st.selectbox('Geography', onehot_encode_geo.categories_[0])
    gender = st.selectbox('Gender', label_endoder_gender.classes_)
    age = st.slider('Age', 18, 92, 30)
    balance = st.number_input('Balance', value=0.0)
    credit_score = st.number_input('Credit Score', value=500)

with col2:
    estimated_salary = st.number_input('Estimated Salary', value=50000)
    tenure = st.slider('Tenure (years)', 0, 10, 2)
    num_of_products = st.slider('Number of Products', 1, 4, 1)
    has_cr_card = st.radio('Has Credit Card', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    is_active_member = st.radio('Is Active Member', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

# Prepare the Input Data
input_data = pd.DataFrame({
    'CreditScore': [float(credit_score)],
    'Gender': [int(label_endoder_gender.transform([gender])[0])],
    'Age': [int(age)],
    'Tenure': [int(tenure)],
    'Balance': [float(balance)],
    'NumOfProducts': [int(num_of_products)],
    'HasCrCard': [int(has_cr_card)],
    'IsActiveMember': [int(is_active_member)],
    'EstimatedSalary': [float(estimated_salary)]
})

# One-hot encode and scale (same as before)
geo_encoded = onehot_encode_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encode_geo.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
input_data = input_data[scaler.feature_names_in_]
input_data_scaled = scaler.transform(input_data)

# Predict Churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

# Display results in a nicer format
st.markdown("---")
st.subheader("Prediction Results")

# Use columns for better layout
result_col1, result_col2 = st.columns(2)

with result_col1:
    st.metric(label="Churn Probability", value=f"{prediction_proba:.2%}")
    
with result_col2:
    if prediction_proba > 0.5:
        st.error('The customer is likely to churn.')
    else:
        st.success('The Customer is not likely to Churn.')

# Optional: Show raw data for debugging
with st.expander("Show raw input data"):
    st.write(input_data)