# import libary 
import streamlit as st
import pandas as pd
import time

# page title
st.set_page_config(
    page_title="Prediksi Stunting Bayi Umur 0-60 Bulan")

hide_style = """
<style>
#MainMenu {visibility: visible;}
footer {visibility: hidden;}
header {visibility: visible;}
</style>
"""
st.markdown(hide_style, unsafe_allow_html=True) 

# Title
st.markdown("<h1 style='text-align: center;'>Prediksi Stunting Bayi Umur 0 - 60 Bulan</h1>", unsafe_allow_html=True)

#Elemen
nama = st.text_input("Masukkan Nama",placeholder='Nama')
umur = st.number_input("Masukkan Umur (bulan)",max_value=60)
jk = st.selectbox("Jenis Kelamin",('Laki-laki','Perempuan'))
tb = st.number_input("Masukkan Tinggi Badan (cm)",max_value=130)

# preprocessing
def normalisasi(x):
    # import data test
    cols = ['umur', 'jk', 'tb']
    df = pd.DataFrame([x],columns=cols)
    data_test = pd.read_csv('data_test.csv')
    data_test = data_test.drop(data_test.columns[0],axis=1)
    # memasukkan data kedalam data test
    data_test = data_test._append(other=df,ignore_index=True)
    # return data_test yang sudah dinormalisasi
    return (data_test)

def knn(x): 
    return joblib.load('modelKNN1.pkl').predict(x)

# Button 
sumbit = st.button("Tes Prediksi")
if sumbit == True:
      if  nama != '' and jk != '' and tb != 0 and umur != 0:
            if jk == 'Laki-laki':
                 jk = 0
            else:
                 jk = 1
            # normalisasi data
            data = normalisasi([umur,jk,tb])
            # prediksi data
            prediksi = knn(data)
            # cek prediksi
            with st.spinner("Tunggu Sebentar..."):
                  if prediksi[-1] == 0:
                        time.sleep(1)
                        st.warning("Hasil Prediksi: "+nama+" Terkena Stunting Parah")
                  elif prediksi [-1] == 1:
                        time.sleep(1)
                        st.warning("Hasil Prediksi: "+nama+" Terkena Stunting")
                  elif prediksi [-1] == 2:
                        time.sleep (1)
                        st.success("Hasil Prediksi: "+nama+" Normal")
                  elif prediksi [-1] == 3:
                        time.sleep (1)
                        st.success("Hasil Prediksi: "+nama+" Tinggi")
      else:
          st.error("Harap Isi Semua Kolom")



