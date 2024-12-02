from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Dữ liệu văn bản mẫu
documents = [
    "Data Science is an interdisciplinary field",
    "Social network analysis is a key tool in Data Science",
    "Graph theory and data are fundamental in network analysis"
]

# TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# Hiển thị kết quả TF-IDF
print("TF-IDF Matrix:")
print(tfidf_matrix.toarray())

# LDA
lda = LatentDirichletAllocation(n_components=2, random_state=0)
lda.fit(tfidf_matrix)
print("Components:")
print(lda.components_)
