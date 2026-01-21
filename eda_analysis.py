import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import re

# -----------------------------
# 1. INPUT: Load Dataset
# -----------------------------
df = pd.read_csv("fake_job_postings.csv")

# Keep only important columns
df = df[['description', 'fraudulent']]
df = df.dropna()

# -----------------------------
# 2. PREPROCESSING
# -----------------------------
def clean_text(text):
    text = text.lower()                                    # lowercase
    text = re.sub(r'http\S+|www.\S+', '', text)            # remove URLs
    text = re.sub(r'[^a-zA-Z ]', ' ', text)                # remove special chars
    text = re.sub(r'\s+', ' ', text).strip()               # remove extra spaces
    return text

df['clean_text'] = df['description'].apply(clean_text)

# -----------------------------
# 3. FEATURE EXTRACTION (CountVectorizer)
# -----------------------------
vectorizer = CountVectorizer(stop_words='english', max_features=50)
X = vectorizer.fit_transform(df['clean_text'])

# Get top 50 common words overall
common_words = vectorizer.get_feature_names_out()


# =========================================================
# ===============    ANALYSIS TASKS    ====================
# =========================================================

# ---------------------------------------------------------
# TASK 1: Visualize fake vs real job posts
# ---------------------------------------------------------
plt.figure(figsize=(6,4))
df['fraudulent'].value_counts().plot(kind='bar')
plt.title("Fake vs Real Job Posts")
plt.xticks([0,1], labels=['Real (0)', 'Fake (1)'])
plt.xlabel("Category")
plt.ylabel("Count")
plt.show()


# ---------------------------------------------------------
# TASK 2: Text Length Analysis
# ---------------------------------------------------------
df['text_length'] = df['clean_text'].apply(lambda x: len(x.split()))

plt.figure(figsize=(7,4))
df['text_length'].hist(bins=40)
plt.title("Distribution of Text Lengths")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.show()


# ANALYSIS: Compare text length of real vs fake
plt.figure(figsize=(7,4))
df[df['fraudulent']==0]['text_length'].hist(alpha=0.5, label='Real', bins=40)
df[df['fraudulent']==1]['text_length'].hist(alpha=0.5, label='Fake', bins=40)
plt.legend()
plt.title("Text Length Comparison: Real vs Fake")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.show()


# ---------------------------------------------------------
# TASK 3: Common words in FAKE job posts
# ---------------------------------------------------------
fake_df = df[df['fraudulent'] == 1]

fake_vectorizer = CountVectorizer(stop_words='english', max_features=20)
fake_matrix = fake_vectorizer.fit_transform(fake_df['clean_text'])

fake_words = fake_vectorizer.get_feature_names_out()
fake_counts = fake_matrix.sum(axis=0).A1

# Plot common words in fake job posts
plt.figure(figsize=(10,5))
plt.bar(fake_words, fake_counts)
plt.xticks(rotation=45)
plt.title("Top Common Words in Fake Job Posts")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.show()
