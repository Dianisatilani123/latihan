import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle

# Memuat dataset
@st.cache_data
def load_data():
    return pd.read_csv('sanitary.csv', delimiter=';')

# Fungsi untuk mempersiapkan data
def preprocess_data(data):
    # Memisahkan data dan label
    X = data.drop(columns=['Outcome'])
    y = data['Outcome']
    
    # Mengubah string ke numerik dengan LabelEncoder
    label_encoders = {}
    for column in ['WHO region', 'Country', 'Residence Area Type']:
        if column in data.columns:
            label_encoders[column] = LabelEncoder()
            X[column] = label_encoders[column].fit_transform(X[column])

    # Melakukan standarisasi pada semua fitur (data)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X, y, X_scaled, scaler, label_encoders

# Fungsi untuk memisahkan data training dan testing
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Fungsi untuk melatih model
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Memuat data
st.title('Model Prediksi Tingkat Sanitasi')
data = load_data()
st.write("Beberapa baris pertama dari DataFrame:")
st.write(data.head())

X, y, X_scaled, scaler, label_encoders = preprocess_data(data)

# Memisahkan data training dan testing
X_train, X_test, y_train, y_test = split_data(X_scaled, y)

# Melatih model
model = train_model(X_train, y_train)

# Menampilkan evaluasi model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
st.write(f"R-squared (R²): {r2:.2f}")
st.write(f"R-squared (R²) dalam persen: {r2*100:.2f}%")

# Input prediksi dari pengguna
st.header('Prediksi Tingkat Sanitasi')
input_tahun = st.number_input('Tahun', min_value=1900, max_value=2100, value=2003)
input_who_region = st.selectbox('WHO region', sorted(data['WHO region'].unique()))
input_country = st.selectbox('Country', sorted(data['Country'].unique()))
input_residence_area = st.selectbox('Residence Area Type', sorted(data['Residence Area Type'].unique()))
input_display_value = st.number_input('Nilai Display Value', min_value=0, max_value=100, value=37)

if st.button('Prediksi'):
    # Mengubah string ke numerik pada data input menggunakan encoder yang sudah dilatih
    input_data_list = [input_tahun, input_who_region, input_country, input_residence_area, input_display_value]
    input_data_list[1] = label_encoders['WHO region'].transform([input_data_list[1]])[0]
    input_data_list[2] = label_encoders['Country'].transform([input_data_list[2]])[0]
    input_data_list[3] = label_encoders['Residence Area Type'].transform([input_data_list[3]])[0]
    
    # Convert input_data_list menjadi tuple
    input_data_tuple = tuple(input_data_list)
    
    # Convert input_data ke numpy array dan buat DataFrame dengan nama kolom asli
    input_data_array = np.array(input_data_tuple).reshape(1, -1)
    input_data_df = pd.DataFrame(input_data_array, columns=X.columns[:5])  # Sesuaikan kolom sesuai dengan input

    # Standarisasi data input
    input_data_scaled = scaler.transform(input_data_df)
    
    # Prediksi
    prediction = model.predict(input_data_scaled)
    st.write('Tingkat Sanitasi Tinggi' if prediction[0] > 0.5 else 'Tingkat Sanitasi Rendah')

# Simpan model
if st.button('Simpan Model'):
    filename = 'sanitary_model.sav'
    pickle.dump(model, open(filename, 'wb'))
    st.write(f"Model disimpan sebagai {filename}")
