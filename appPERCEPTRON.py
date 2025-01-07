import streamlit as st
import numpy as np
import joblib
import os

# Load saved models
model = joblib.load('perceptron_model.pkl')
scaler = joblib.load('scaler_perceptron.pkl')
label_encoder = joblib.load('label_encoder_Perceptron.pkl')

# Streamlit App
def main():
    st.title("Perediksi Fruit Perceptron")
    st.write("Memprediksi data baru dengan algoritma PERCEPTRON")

    # Input features
    st.header("Masukkan Data Fruit")
    diameter = st.number_input("diameter", value=0.0, format="%.2f")
    weight = st.number_input("weight", value=0.0, format="%.2f")
    red = st.number_input("red", value=0.0, format="%.2f")
    green = st.number_input("green", value=0.0, format="%.2f")
    blue = st.number_input("blue", value=0.0, format="%.2f")

    # Collect input into a single array
    input_data = np.array([[diameter, weight, red, green, blue]])

    if st.button("Predict Class"):
        try:
            # Scale input data
            scaled_data = scaler.transform(input_data)

            # Make prediction
            prediction = model.predict(scaled_data)

            # Decode the predicted class
            predicted_class = label_encoder.inverse_transform(prediction)[0]

            st.success(f"Prediksi : {predicted_class}")
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
