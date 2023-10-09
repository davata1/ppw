import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

df = pd.read_csv('https://raw.githubusercontent.com/davata1/ppw/main/DataPTAInformatikaLabel.csv',delimiter=';')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.head()

# Sekarang df berisi data dari file CSV
print(df.head())  # Cetak beberapa baris pertama dari DataFrame untuk memastikan data berhasil dimuat

# Fungsi untuk melakukan LDA dan menampilkan topik
def lakukan_lda(data_teks, num_topics=3):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data_teks)
    
    # Model LDA
    model_lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    matriks_topik_lda = model_lda.fit_transform(X)
    
    topik_lda = []
    for i, dokumen in enumerate(data_teks):
        probabilitas_topik = matriks_topik_lda[i]
        top_topik = probabilitas_topik.argsort()[-2:][::-1]  # Dapatkan 2 topik teratas untuk setiap dokumen
        topik_lda.append(top_topik)
    return topik_lda

# Aplikasi Streamlit
st.title("Pemodelan Topik LDA")
st.write("Temukan topik dalam data teks Anda menggunakan Latent Dirichlet Allocation (LDA)")

# Menampilkan input data teks
data_teks = st.text_area("Masukkan data teks Anda (satu dokumen per baris)", height=200)

if data_teks:
    data_teks = data_teks.split('\n')  # Memisahkan input menjadi dokumen individu
    jumlah_topik = st.slider("Jumlah Topik", min_value=1, max_value=10, value=3, step=1)
    topik_lda = lakukan_lda(data_teks, jumlah_topik)
    
    st.subheader("Topik Teratas untuk Setiap Dokumen:")
    for i, dokumen_topik in enumerate(topik_lda):
        st.write(f"Dokumen {i + 1}: Topik - {dokumen_topik[0] + 1}, {dokumen_topik[1] + 1}")
