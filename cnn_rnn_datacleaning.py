import re
import pandas as pd
import nltk
import spacy
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os

# Install spaCy model if not installed
os.system("python -m spacy download en_core_web_sm")

# Download NLTK stopwords
nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words('english'))

# Load spaCy English model with disabled ner and parser
nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser'])

# 1. Text cleaning function
def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+|www\S+', ' ', text)  # remove URLs
    text = re.sub(r'\S+@\S+', ' ', text)  # remove emails
    text = re.sub(r'\+?\d[\d -]{8,12}\d', ' ', text)  # remove phone numbers
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)  # remove emojis
    text = re.sub(r'[^\w\s]', ' ', text)  # keep only alphabets, numbers, spaces
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra spaces
    return text

# 2. Load dataset (ensure correct filename and path)
df = pd.read_csv("fake_job_postings.csv")

# Use important columns if present
columns_to_keep = ["title", "location", "description", "requirements", "salary_range"]
columns_present = [col for col in columns_to_keep if col in df.columns]
df = df[columns_present]

# Data cleaning steps
df = df.dropna(subset=["description"])
df = df.drop_duplicates()
df = df[df["description"].str.len() > 20]
df = df[df["description"].str.contains('[A-Za-z]', regex=True)]
df["clean_text"] = df["description"].apply(clean_text)

# Save cleaned data for reference
df.to_csv("cleaned_jobs.csv", index=False)
print("✔ Cleaning completed and saved to cleaned_jobs.csv")

# 3. Lemmatization removing stopwords
def lemmatize(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if token.text not in stopwords])

df["processed_text"] = df["clean_text"].apply(lemmatize)

# 4. Tokenization and padding
texts = df["processed_text"].tolist()
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=200, padding='post')

# 5. Create labels (update as per your dataset, here example uses 'fraudulent' column if present)
if "fraudulent" in df.columns:
    labels = df["fraudulent"].values.astype(int)
else:
    # Dummy labels: mark "intern" in title as fake (replace with your actual label logic)
    labels = df["title"].apply(lambda x: 1 if "intern" in str(x).lower() else 0).values

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(padded, labels, test_size=0.2, random_state=42)

# 6. CNN model architecture
cnn_model = Sequential([
    Embedding(10000, 64, input_length=200),
    Conv1D(64, 3, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

cnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
print("\nTraining CNN model...")
cnn_model.fit(X_train, y_train, epochs=3, batch_size=32, validation_data=(X_test, y_test))

cnn_model.save("cnn_model.h5")
print(" CNN model saved as cnn_model.h5")

# 7. RNN (LSTM) model architecture
rnn_model = Sequential([
    Embedding(10000, 64, input_length=200),
    LSTM(64, return_sequences=False),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

rnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
print("\nTraining RNN (LSTM) model...")
rnn_model.fit(X_train, y_train, epochs=3, batch_size=32, validation_data=(X_test, y_test))

rnn_model.save("rnn_model.h5")
print(" RNN model saved as rnn_model.h5")

print("\n Training complete! Models saved: cnn_model.h5, rnn_model.h5")
