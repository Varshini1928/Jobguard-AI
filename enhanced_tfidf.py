
from sklearn.feature_extraction.text import TfidfVectorizer

# Preprocessed text column from your cleaned dataset
texts = ["processed_text"]

tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english",
    ngram_range=(1,2)   # unigrams + bigrams
)

X = tfidf.fit_transform(texts)

print("TF-IDF Shape:", X.shape)
