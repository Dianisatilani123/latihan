import numpy as np
import pandas as pd
from sklearn.base import is_classifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Memuat dataset dengan delimiter ';'
sanitary_dataset = pd.read_csv('sanitary.csv', delimiter=';')

# Menampilkan beberapa baris pertama dari DataFrame untuk memastikan data dibaca dengan benar
print("Beberapa baris pertama dari DataFrame:")
print(sanitary_dataset.head())

# Menampilkan nama kolom yang ada
print("\nKolom yang tersedia dalam DataFrame:", sanitary_dataset.columns)

# Memeriksa apakah kolom 'Outcome' ada
if 'Outcome' in sanitary_dataset.columns:
    outcome_counts = sanitary_dataset['Outcome'].value_counts()
    print("\nValue count of Outcome:")
    print(outcome_counts)
else:
    print("Kolom 'Outcome' tidak ditemukan dalam DataFrame.")

# Memisahkan data dan label
X = sanitary_dataset.drop(columns=['Outcome'])
y = sanitary_dataset['Outcome']

# Menampilkan jumlah data untuk setiap kelompok
print("\nJumlah data:", len(X))
print("Jumlah label:", len(y))

# Menampilkan beberapa baris pertama dari setiap kelompok
print("\nBeberapa baris pertama dari data:")
print(X.head())

print("\nBeberapa baris pertama dari label:")
print(y.head())

# Mengubah string ke numerik dengan LabelEncoder
label_encoders = {}
for column in ['WHO region', 'Country', 'Residence Area Type']:
    if column in sanitary_dataset.columns:
        label_encoders[column] = LabelEncoder()
        X[column] = label_encoders[column].fit_transform(X[column])

# Melakukan standarisasi pada semua fitur (data)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert scaled array back to DataFrame dengan nama kolom asli
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Menampilkan beberapa baris pertama dari DataFrame setelah standarisasi
print("\nBeberapa baris pertama dari DataFrame setelah standarisasi:")
print(X_scaled_df.head())

# Memisahkan data training dan testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)

# Menampilkan informasi tentang data yang sudah dipisahkan
print("\nJumlah data training (X_train):", len(X_train))
print("Jumlah data testing (X_test):", len(X_test))
print("Jumlah label training (y_train):", len(y_train))
print("Jumlah label testing (y_test):", len(y_test))

# Pastikan ukuran X_train dan y_train sama, serta X_test dan y_test sama
assert len(X_train) == len(y_train), "Jumlah baris X_train dan y_train tidak sama!"
assert len(X_test) == len(y_test), "Jumlah baris X_test dan y_test tidak sama!"

# Menampilkan beberapa baris pertama dari data training dan data testing
print("\nBeberapa baris pertama dari data training (X_train):")
print(X_train[:5])

print("\nBeberapa baris pertama dari data testing (X_test):")
print(X_test[:5])

# split data train dan data test
def split_data(data):
    X = data.drop(columns=["Display Value", "Numeric"])  # Hapus fitur "Numeric"
    y = data["Display Value"]

    test_size = 0.2  # Tentukan ukuran test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Menghitung persentase data training dan testing
    train_percentage = len(X_train) / len(data) * 100
    test_percentage = len(X_test) / len(data) * 100

    np.std.write(f"Persentase data training: {train_percentage:.2f}%")
    np.std.write(f"Persentase data testing: {test_percentage:.2f}%")

    return X_train, X_test,y_train,y_test

# Membuat model regresi linear
regression_model = LinearRegression()

# Melatih model regresi dengan data training
regression_model.fit(X_train, y_train)

# Memprediksi hasil dengan data testing
y_pred = regression_model.predict(X_test)

# Evaluasi performa model menggunakan Mean Squared Error (MSE), Mean Absolute Error (MAE), dan R-squared (R²)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R²): {r2:.2f}")
print(f"R-squared (R²) dalam persen: {r2*100:.2f}%")

# Model prediksi
input_data = (2003, 'Europe', 'Albania', 'Rural', 37, 3679997)  

# Mengubah string ke numerik pada data input menggunakan encoder yang sudah dilatih
input_data_list = list(input_data)
input_data_list[1] = label_encoders['WHO region'].transform([input_data_list[1]])[0]
input_data_list[2] = label_encoders['Country'].transform([input_data_list[2]])[0]
input_data_list[3] = label_encoders['Residence Area Type'].transform([input_data_list[3]])[0]

# Convert input_data_list menjadi tuple
input_data_tuple = tuple(input_data_list)

# Convert input_data ke numpy array dan buat DataFrame dengan nama kolom asli
input_data_array = np.array(input_data_tuple).reshape(1, -1)
input_data_df = pd.DataFrame(input_data_array, columns=X.columns)

# Standarisasi data input
input_data_scaled = scaler.transform(input_data_df)

print(input_data_scaled)

prediction = regression_model.predict(input_data_scaled)
print(prediction)

if prediction[0] == 0:
    print('Tingkat Sanitasi Rendah')
else:
    print('Tingkat Sanitasi Tinggi')


# simpan model
import pickle

filename = 'sanitary_model.sav'
pickle.dump(regression_model, open(filename, 'wb'))