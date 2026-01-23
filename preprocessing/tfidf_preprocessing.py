import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------------------------------------------
# 1. LOAD CLEANED DATASET
# -------------------------------------------------------
df = pd.read_csv("cleaned_jobs.csv")

print("Dataset columns:", list(df.columns))

# -------------------------------------------------------
# 2. SIMPLE PREPROCESSING FUNCTION
#    (use existing clean_text column if available)
# -------------------------------------------------------
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

if "clean_text" in df.columns:
    text_col = "clean_text"
else:
    text_col = "description"

print(f"Processing column: '{text_col}'")
df["processed_text"] = df[text_col].astype(str).apply(preprocess_text)

# -------------------------------------------------------
# 3. TF-IDF VECTORISATION
# -------------------------------------------------------
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df["processed_text"])

# -------------------------------------------------------
# 4. SAVE OUTPUTS
# -------------------------------------------------------
pd.DataFrame(X.toarray(),
             columns=tfidf.get_feature_names_out()).to_csv("tfidf_features.csv",
                                                          index=False)
df.to_csv("final_preprocessed_dataset.csv", index=False)

print("Preprocessing Completed!")
print("Saved: final_preprocessed_dataset.csv")
print("TF-IDF Features Saved: tfidf_features.csv")
