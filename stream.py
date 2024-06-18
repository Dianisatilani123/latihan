import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib
import streamlit as st
import time

# Load data
data = pd.read_csv('data_balita.csv')

# Preprocessing
y_class = data['sg']
y = y_class.values.tolist()
x = data.drop(columns='sg')
scaler = MinMaxScaler()
scaled = scaler.fit_transform(x)
nama_fitur = x.columns.copy()
scaled_fitur = pd.DataFrame(scaled, columns=nama_fitur)
joblib.dump(scaler, 'norm.sav')

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
x_test.to_csv('data_test.csv')

# Train KNN model
k_range = range(1, 31)
scores = {}
scores_list = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    filenameKNN = 'odelKNN' + str(1) + '.pkl'
    joblib.dump(knn, filenameKNN)
    y_pred = knn.predict(x_test)
    scores[k] = accuracy_score(y_test, y_pred)
    scores_list.append(accuracy_score(y_test, y_pred))

# Get best K value
best_k = scores_list.index(max(scores_list)) + 1
print('Best K value:', best_k)

# Evaluate model
knn = joblib.load('modelKNN1.pkl')
x_train_pred = knn.predict(x_train)
accuracy = accuracy_score(y_train, x_train_pred)
print('Akurasi data training:', accuracy)
x_test_pred = knn.predict(x_test)
accuracy_test = accuracy_score(y_test, x_test_pred)
print('Akurasi data testing:', accuracy_test)

# Streamlit app
st.set_page_config(page_title="Prediksi Stunting Bayi Umur 0-60 Bulan")
hide_style = """
<style>
#MainMenu {visibility: visible;}
footer {visibility: hidden;}
header {visibility: visible;}
</style>
"""
st.markdown(hide_style, unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>Prediksi Stunting Bayi Umur 0 - 60 Bulan</h1>", unsafe_allow_html=True)

nama = st.text_input("Masukkan Nama", placeholder='Nama')
umur = st.number_input("Masukkan Umur (bulan)", max_value=60)
jk = st.selectbox("Jenis Kelamin", ('Laki-laki', 'Perempuan'))
tb = st.number_input("Masukkan Tinggi Badan (cm)", max_value=130)

def normalisasi(x):
    cols = ['umur', 'jk', 'tb']
    df = pd.DataFrame([x], columns=cols)
    data_test = pd.read_csv('data_test.csv')
    data_test = data_test.drop(data_test.columns[0], axis=1)
    data_test = data_test._append(other=df, ignore_index=True)
    scaler = joblib.load('norm.sav')
    return scaler.transform(data_test)

def knn(x):
    return joblib.load('modelKNN1.pkl').predict(x)

sumbit = st.button("Tes Prediksi")
if sumbit == True:
    if nama!= '' and jk!= '' and tb!= 0 and umur!= 0:
        if jk == 'Laki-laki':
            jk = 0
        else:
            jk = 1
        data = normalisasi([umur, jk, tb])
        prediksi = knn(data)
        with st.spinner("Tunggu Sebentar..."):
            if prediksi[-1] == 0:
                time.sleep(1)
                st.warning("Hasil Prediksi: " + nama + " Terkena Stunting Parah")
            elif prediksi[-1] == 1:
                time.sleep(1)
                st.warning("Hasil Prediksi: " + nama + " Terkena Stunting")
            elif prediksi[-1] == 2:
                time.sleep(1)
                st.success("Hasil Prediksi: " + nama + " Normal")
            elif prediksi[-1] == 3:
                time.sleep(1)
                st.success("Hasil Prediksi: " + nama + " Tinggi")
    else:
        st.error("Harap Isi Semua Kolom")
