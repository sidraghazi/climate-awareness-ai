import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt

# =========================
# Load Dataset
# =========================
df = pd.read_csv("twitter_sentiment_data.csv")

# =========================
# Map labels to climate awareness classes
# =========================
label_mapping = {
    1: "High Awareness (Pro)",
    0: "Moderate Awareness (Neutral)",
    -1: "Low Awareness (Anti)",
    2: "Informational Awareness (News)"
}

df["climate_awareness"] = df["sentiment"].map(label_mapping)

# Check label distribution
print("Class distribution:\n", df["climate_awareness"].value_counts(), "\n")

# =========================
# Text Cleaning
# =========================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)   # remove URLs
    text = re.sub(r"@\w+", "", text)             # remove mentions
    text = re.sub(r"#", "", text)                # remove hashtag symbol
    text = re.sub(r"[^a-z\s]", "", text)         # keep letters only
    text = re.sub(r"\s+", " ", text).strip()     # remove extra spaces
    return text

df["clean_message"] = df["message"].astype(str).apply(clean_text)

# =========================
# Train-Test Split
# =========================
X = df["clean_message"]
y = df["climate_awareness"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# TF-IDF Vectorization
# =========================
tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=5000,
    ngram_range=(1, 2)
)


X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# =========================
# Train Model
# =========================
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# =========================
# Evaluate Model
# =========================
y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred), "\n")

print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# =========================
# Confusion Matrix
# =========================
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

plt.figure(figsize=(7, 6))
plt.imshow(cm)
plt.title("Confusion Matrix – Climate Awareness Classification")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(range(len(model.classes_)), model.classes_, rotation=45)
plt.yticks(range(len(model.classes_)), model.classes_)
plt.colorbar()
plt.tight_layout()
plt.show()

# =========================
# Test Custom Input
# =========================
sample_text = [
    "New UN report highlights rising global temperatures and sea levels"
]

sample_clean = [clean_text(sample_text[0])]
sample_tfidf = tfidf.transform(sample_clean)

prediction = model.predict(sample_tfidf)
print("Sample Prediction:", prediction[0])


