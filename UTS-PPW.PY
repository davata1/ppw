import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

st.header("UTS PPW")
st.subheader("Mengambil Data CSV pada Github")
st.write("**pada repo Zey21/dataset/**")
st.text("load data(DataStemming.csv) csv yang sudah berhasil di stemming")

##Load data
df = pd.read_csv('https://raw.githubusercontent.com/davata1/ppw/main/DataSteaming.csv')
df['Abstrak'].fillna('', inplace=True)

#Ektraksi Fitur
count_vectorizer = CountVectorizer()
X_count = count_vectorizer.fit_transform(np.array(df['Abstrak']))
terms_count = count_vectorizer.get_feature_names_out()
df_countvect = pd.DataFrame(data=X_count.toarray(), columns=terms_count)

##LDA Modelling
lda_model = LatentDirichletAllocation(n_components=3, doc_topic_prior=0.1, topic_word_prior=0.2, random_state=42)
lda_model.fit(X_count)

doc_topic_proportions = lda_model.transform(X_count)

for i, doc in enumerate(df['Abstrak']):
    st.text(f"Dokumen {i+1}:")
    for j, topic_prob in enumerate(doc_topic_proportions[i]):
        st.text(f"Topik {j+1}: {topic_prob:.4f}")
    st.text("")

topic_word_distributions = lda_model.components_

feature_names = count_vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(topic_word_distributions):
    top_words_idx = topic.argsort()[::-1][:10]  # Ambil 10 kata teratas
    top_words = [feature_names[i] for i in top_words_idx]
    st.text(f"Topik {topic_idx+1}:")
    st.text(", ".join(top_words))
    st.text("")
