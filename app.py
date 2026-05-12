import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Car Price Predictor", layout="centered")
st.title("🚗 Used Car Price & Quality Predictor")
st.markdown("**MSc Data Science Project**")

# Load artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load('best_car_price_model.pkl')
    scaler = joblib.load('scaler.pkl')
    encoder = joblib.load('target_encoder.pkl')
    features = joblib.load('feature_columns.pkl')
    return model, scaler, encoder, features

model, scaler, encoder, features = load_artifacts()

# Sidebar inputs
st.sidebar.header("Car Details")
car_age = st.sidebar.slider("Car Age (years)", 0, 30, 5)
milage = st.sidebar.number_input("Mileage", 0, 300000, 45000)
brand = st.sidebar.selectbox("Brand", ["Toyota", "Honda", "Ford", "BMW", "Mercedes-Benz", "Audi", "Lexus", "Tesla", "Other"])
fuel_type = st.sidebar.selectbox("Fuel Type", ["Gasoline", "Hybrid", "Electric", "Diesel"])
accident = st.sidebar.selectbox("Accident", ["None reported", "At least 1 accident or damage reported"])
clean_title = st.sidebar.selectbox("Clean Title", ["Yes", "No"])

# Create input DataFrame
input_df = pd.DataFrame(0, index=[0], columns=features)

input_df['car_age'] = car_age
input_df['milage_log'] = np.log1p(milage)

# Apply target encoding
temp = pd.DataFrame({'brand': [brand], 'model': ['Unknown']})
temp_encoded = encoder.transform(temp)
input_df['brand'] = temp_encoded['brand'].iloc[0]

# One-hot
if f'fuel_type_{fuel_type}' in features:
    input_df[f'fuel_type_{fuel_type}'] = 1
if f'accident_{accident}' in features:
    input_df[f'accident_{accident}'] = 1
if f'clean_title_{clean_title}' in features:
    input_df[f'clean_title_{clean_title}'] = 1

# Predict
input_scaled = scaler.transform(input_df)
price_log = model.predict(input_scaled)
predicted_price = np.expm1(price_log[0])

st.success(f"**Predicted Price: ${predicted_price:,.0f}**")

# Simple quality classification
if predicted_price > 45000:
    st.success("**Quality: Premium** 🟢")
elif predicted_price > 25000:
    st.warning("**Quality: Good** 🟡")
else:
    st.error("**Quality: Average** 🔴")
