import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import is_classifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle

# Fungsi untuk memuat data
def load_data():
    return pd.read_csv('sanitary.csv', delimiter=';')

# Fungsi untuk melatih model
def train_model(data):
    X = data[['Year', 'Country', 'Residence Area Type', 'Display Value']]
    y = data['Outcome']

    label_encoders = {}
    for column in ['Country', 'Residence Area Type']:
        label_encoders[column] = LabelEncoder()
        X[column] = label_encoders[column].fit_transform(X[column])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)
    
    y_pred = regression_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R-squared (R²): {r2:.2f}")
    print(f"R-squared (R²) dalam persen: {r2*100:.2f}%")

    return regression_model, scaler, label_encoders

# Fungsi untuk memprediksi dengan model terlatih
def predict(model, scaler, label_encoders, input_data):
    input_data_list = list(input_data)
    input_data_list[1] = label_encoders['Country'].transform([input_data_list[1]])[0]
    input_data_list[2] = label_encoders['Residence Area Type'].transform([input_data_list[2]])[0]
    
    input_data_array = np.array(input_data_list).reshape(1, -1)
    input_data_scaled = scaler.transform(input_data_array)
    
    prediction = model.predict(input_data_scaled)
    return prediction[0]

# Fungsi utama untuk aplikasi Streamlit
def main():
    st.title("Prediksi Tingkat Sanitasi")

    # Memuat data dan melatih model
    data = load_data()
    model, scaler, label_encoders = train_model(data)

    # Input pengguna
    st.header("Input Data")
    year = st.number_input("Tahun", min_value=2000, max_value=2023, value=2020)
    country = st.selectbox("Negara", data['Country'].unique())
    residence_area = st.selectbox("Area Tempat Tinggal", data['Residence Area Type'].unique())
    display_value = st.number_input("Nilai Display", min_value=0.0, max_value=100.0, value=50.0)

    # Melakukan prediksi saat tombol diklik
    if st.button("Prediksi"):
        input_data = (year, country, residence_area, display_value)
        result = predict(model, scaler, label_encoders, input_data)
        
        if result < 0.5:
            st.write('Tingkat Sanitasi Rendah')
        else:
            st.write('Tingkat Sanitasi Tinggi')

# Menjalankan aplikasi Streamlit
if __name__ == "__main__":
    main()
