import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pickle

# --------------------------------------------
# 1. LOAD PREPROCESSED DATA
# --------------------------------------------
df = pd.read_csv("final_preprocessed_dataset.csv")
print("Columns:", df.columns.tolist())

# We already have processed_text
TEXT_COL = "processed_text"

# --------------------------------------------
# 2. CREATE LABEL COLUMN
#    0 = Real, 1 = Fake
# --------------------------------------------
if "fraudulent" not in df.columns:
    df["fraudulent"] = 0
    # mark first 100 rows as fake (you can change this rule later)
    n_fake = min(100, len(df))
    df.loc[:n_fake-1, "fraudulent"] = 1
    print(f"Created 'fraudulent' label: {n_fake} fake, {len(df)-n_fake} real")

LABEL_COL = "fraudulent"

X_text = df[TEXT_COL].astype(str)
y = df[LABEL_COL]

# --------------------------------------------
# 3. FEATURE EXTRACTION (TF-IDF)
# --------------------------------------------
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(X_text)

# Save TF-IDF vectorizer for Flask
with open("tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f)

# --------------------------------------------
# 4. TRAIN-TEST SPLIT
# --------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------------------------------
# 5. TRAIN MODELS
# --------------------------------------------
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# --------------------------------------------
# 6. PREDICTIONS & METRICS
# --------------------------------------------
log_pred = log_model.predict(X_test)
nb_pred = nb_model.predict(X_test)

log_acc = accuracy_score(y_test, log_pred)
nb_acc  = accuracy_score(y_test, nb_pred)

print("\nLogistic Regression Accuracy :", log_acc)
print("Naive Bayes Accuracy        :", nb_acc)

print("\n--- Logistic Regression Report ---")
print(classification_report(y_test, log_pred))

print("\n--- Naive Bayes Report ---")
print(classification_report(y_test, nb_pred))

# --------------------------------------------
# 7. SAVE MODELS FOR FLASK API
# --------------------------------------------
with open("log_model.pkl", "wb") as f:
    pickle.dump(log_model, f)

with open("nb_model.pkl", "wb") as f:
    pickle.dump(nb_model, f)

print("\nModels saved as:")
print(" → tfidf.pkl")
print(" → log_model.pkl")
print(" → nb_model.pkl")
