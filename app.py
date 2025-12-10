import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

# ============================
# LOAD DATASET
# ============================
df = pd.read_csv("netflix_titles.csv")

# Pastikan kolom yang dibutuhkan ada
df["genre_main"] = df["listed_in"].apply(lambda x: x.split(",")[0] if isinstance(x, str) else "")
df["description"] = df["description"].fillna("")

# ============================
# MODEL REKOMENDASI
# ============================
df["combined"] = df["genre_main"] + " " + df["description"]

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["combined"])


def rekomendasi_film(judul):
    if judul not in df["title"].values:
        return None

    idx = df[df["title"] == judul].index[0]

    similarity_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    df_sim = df.copy()
    df_sim["similarity"] = similarity_scores

    df_sim = df_sim.sort_values(by="similarity", ascending=False).reset_index(drop=True)
    df_sim = df_sim.iloc[1:6]  # top 5 rekomendasi

    def kategori(nilai):
        if nilai >= 0.40:
            return "HIGH"
        elif nilai >= 0.20:
            return "MEDIUM"
        else:
            return "LOW"

    df_sim["Rekomendasi"] = df_sim["similarity"].apply(kategori)

    return df_sim[["title", "genre_main", "similarity", "Rekomendasi"]]


# ============================
# MODEL PREDIKSI MOVIE / TV SHOW
# ============================
rating_encoder = LabelEncoder()
genre_encoder = LabelEncoder()

df["rating_clean"] = rating_encoder.fit_transform(df["rating"].astype(str))
df["genre_clean"] = genre_encoder.fit_transform(df["genre_main"].astype(str))

# Fitur durasi (Movie = menit, TV = jumlah episode)
df["durasi"] = df["duration"].str.extract('(\d+)').astype(float)

# Isi NaN
df["durasi"] = df["durasi"].fillna(df["durasi"].median())

# Target
df["type_num"] = df["type"].map({"Movie": 0, "TV Show": 1})

# Dummy model (logistic regression)
from sklearn.linear_model import LogisticRegression

X = df[["rating_clean", "durasi", "genre_clean"]]
y = df["type_num"]

model = LogisticRegression()
model.fit(X, y)


def prediksi_type(rating, durasi, genre):
    try:
        rating_enc = rating_encoder.transform([rating])[0]
        genre_enc = genre_encoder.transform([genre])[0]
    except:
        return None

    pred = model.predict([[rating_enc, durasi, genre_enc]])[0]
    return "TV Show" if pred == 1 else "Movie"


# ============================
# STREAMLIT UI
# ============================
st.set_page_config(page_title="Movie Recommendation App", layout="wide")

st.markdown(
    """
    <h1 style='text-align: center;'>🎬 Movie Recommendation App (Netflix Dataset)</h1>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# BAGIAN 1 — REKOMENDASI FILM
# -------------------------------------------------
st.subheader("🔍 Cari Rekomendasi Film")

judul = st.text_input("Masukkan judul film:")

if judul:
    result = rekomendasi_film(judul)

    if result is None:
        st.error("Judul tidak ditemukan dalam dataset!")
    else:
        st.success(f"Hasil Rekomendasi Untuk: **{judul}**")
        st.dataframe(result)

        top = result.iloc[0]["Rekomendasi"]
        if top == "HIGH":
            st.success("⭐ Tingkat Rekomendasi: **HIGH**")
        elif top == "MEDIUM":
            st.warning("🟡 Tingkat Rekomendasi: **MEDIUM**")
        else:
            st.error("🔴 Tingkat Rekomendasi: **LOW**")


# -------------------------------------------------
# BAGIAN 2 — PREDIKSI MOVIE / TV SHOW
# -------------------------------------------------
st.subheader("📌 Prediksi Jenis Konten (Movie / TV Show)")

rating_in = st.text_input("Rating (misal: PG-13, TV-MA, R):")
durasi_in = st.number_input("Durasi (angka saja):", min_value=1, value=1)
genre_in = st.text_input("Genre (misal: Dramas, Action):")

if st.button("Prediksi"):
    hasil = prediksi_type(rating_in, durasi_in, genre_in)
    if hasil is None:
        st.error("Input tidak dikenali. Pastikan rating/genre sesuai data!")
    else:
        if hasil == "Movie":
            st.success(f"🎬 Jenis Konten: **{hasil}**")
        else:
            st.warning(f"📺 Jenis Konten: **{hasil}**")
