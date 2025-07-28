import streamlit as st
import pandas as pd
import joblib
from surprise import SVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Veri ve model önbellekleme
@st.cache_data
def load_data():
    books = pd.read_csv("books.csv")
    ratings = pd.read_csv("ratings.csv")
    tags = pd.read_csv("tags.csv")
    book_tags = pd.read_csv("book_tags.csv")

    book_tags_merged = pd.merge(book_tags, tags, on="tag_id", how="left")
    book_tags_grouped = book_tags_merged.groupby("goodreads_book_id")["tag_name"].apply(lambda x: ' '.join(x)).reset_index()
    books_with_tags = pd.merge(books, book_tags_grouped, on="goodreads_book_id", how="left")

    return books, ratings, books_with_tags

@st.cache_data
def prepare_tfidf_cosine(books_with_tags):
    books_with_tags['tag_name'] = books_with_tags['tag_name'].fillna("")
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(books_with_tags['tag_name'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return tfidf_matrix, cosine_sim

@st.cache_resource
def load_model():
    return joblib.load("svd_model.pkl")

# Veriyi ve modeli yükle
books, ratings, books_with_tags = load_data()
tfidf_matrix, cosine_sim = prepare_tfidf_cosine(books_with_tags)
svd_model = load_model()

# Aykırı kitapları çıkar (5000'den fazla puan almış olanlar)
book_rating_counts = ratings['book_id'].value_counts()
popular_books = book_rating_counts[book_rating_counts < 5000].index
filtered_ratings = ratings[ratings['book_id'].isin(popular_books)]

# Yardımcı fonksiyonlar
def kitap_isimden_id_al(secimler, kitaplar_df):
    kitaplar_df['title_lower'] = kitaplar_df['title'].str.lower()
    secimler = [s.lower() for s in secimler]
    book_ids = kitaplar_df[kitaplar_df['title_lower'].isin(secimler)]['book_id'].unique().tolist()
    return book_ids

def kitap_oner_secim_uzerinden(book_ids, ratings_df, books_df, top_n=10, min_rating=4):
    begenenler = ratings_df[
        (ratings_df['book_id'].isin(book_ids)) & 
        (ratings_df['rating'] >= min_rating)
    ]['user_id'].unique()

    diger_kitaplar = ratings_df[
        (ratings_df['user_id'].isin(begenenler)) &
        (~ratings_df['book_id'].isin(book_ids)) &
        (ratings_df['rating'] >= min_rating)
    ]

    kitap_onerileri = diger_kitaplar['book_id'].value_counts().head(top_n).index.tolist()
    sonuc = books_with_tags[books_with_tags['book_id'].isin(kitap_onerileri)][['title', 'authors', 'tag_name']].drop_duplicates()
    return sonuc.head(top_n)

def kitap_icerik_tabanli_oner(kitap_adi, df=books_with_tags, cosine_sim=cosine_sim, top_n=10):
    df = df.reset_index()
    indices = pd.Series(df.index, index=df['title'].str.lower())
    kitap_adi = kitap_adi.lower()
    if kitap_adi not in indices:
        return pd.DataFrame([{'title': 'Kitap bulunamadı'}])
    idx = indices[kitap_adi]
    sim_scores = list(enumerate(cosine_sim[idx].tolist()))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    kitap_indices = [i[0] for i in sim_scores]
    return df.loc[kitap_indices, ['title', 'authors', 'tag_name']]

# Streamlit Arayüzü
st.title("📚 Kitap Öneri Sistemi")

secilen_yontem = st.sidebar.radio("Öneri Yöntemi Seç:", ["Türe Göre Öneri", "Beğenilere Göre Öneri", "SVD Modeline Göre Öneri"])

kitap_listesi = books['title'].dropna().unique().tolist()

if secilen_yontem == "Türe Göre Öneri":
    secim = st.multiselect("Beğendiğiniz kitap(lar)ı seçin:", kitap_listesi)
    if st.button("📖 Benzer Kitapları Göster"):
        if len(secim) == 0:
            st.warning("Lütfen en az bir kitap seçin.")
        else:
            sonuc = kitap_icerik_tabanli_oner(secim[0])
            st.write("🔎 Benzer içerikli kitaplar:")
            st.dataframe(sonuc)

elif secilen_yontem == "Beğenilere Göre Öneri":
    secim = st.multiselect("Beğendiğiniz kitap(lar)ı seçin:", kitap_listesi)
    if st.button("📖 Kullanıcı Tabanlı Önerileri Göster"):
        if len(secim) == 0:
            st.warning("Lütfen en az bir kitap seçin.")
        else:
            ids = kitap_isimden_id_al(secim, books)
            sonuc = kitap_oner_secim_uzerinden(ids, filtered_ratings, books)
            st.write("👥 Benzer kullanıcıların önerdiği kitaplar:")
            st.dataframe(sonuc)

elif secilen_yontem == "SVD Modeline Göre Öneri":
    user_ids = filtered_ratings['user_id'].unique().tolist()
    secilen_user = st.selectbox("Kullanıcı ID Seç:", user_ids)

    if st.button("🔮 Tahmini Önerileri Göster"):
        kullanici_kitaplar = filtered_ratings[filtered_ratings['user_id'] == secilen_user]['book_id'].tolist()
        tum_kitaplar = filtered_ratings['book_id'].unique().tolist()
        onerilebilecekler = list(set(tum_kitaplar) - set(kullanici_kitaplar))

        tahminler = [(book_id, svd_model.predict(secilen_user, book_id).est) for book_id in onerilebilecekler]
        en_iyi = sorted(tahminler, key=lambda x: x[1], reverse=True)[:10]
        en_iyi_ids = [x[0] for x in en_iyi]

        sonuc = books_with_tags[books_with_tags['book_id'].isin(en_iyi_ids)][['title', 'authors', 'tag_name']].drop_duplicates()
        st.write("🔮 SVD Modeline Göre Öneriler:")
        st.dataframe(sonuc)
