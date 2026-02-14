# import pandas as pd
# import pickle
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score

# # -------------------------
# # Load datasets
# # -------------------------
# df_fake = pd.read_csv("news_dataset.csv/Fake.csv")
# df_true = pd.read_csv("news_dataset.csv/True.csv")

# # -------------------------
# # Add labels
# # -------------------------
# df_fake["label"] = 0   # Fake News
# df_true["label"] = 1   # Real News

# # -------------------------
# # Combine datasets
# # -------------------------
# df = pd.concat([df_fake, df_true], axis=0)
# df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# print("Class Distribution:")
# print(df["label"].value_counts())

# # -------------------------
# # Combine title + text
# # -------------------------
# df["content"] = df["title"] + " " + df["text"]

# # -------------------------
# # Clean text
# # -------------------------
# df["content"] = df["content"].str.lower()
# df["content"] = df["content"].str.replace("[^a-zA-Z ]", "", regex=True)

# # -------------------------
# # Features and labels
# # -------------------------
# X = df["content"]
# y = df["label"]

# # -------------------------
# # TF-IDF Vectorization
# # -------------------------
# vectorizer = TfidfVectorizer(
#     stop_words="english",
#     max_df=0.7,
#     ngram_range=(1, 2)
# )

# X_vectorized = vectorizer.fit_transform(X)

# # -------------------------
# # Train/Test split
# # -------------------------
# X_train, X_test, y_train, y_test = train_test_split(
#     X_vectorized, y, test_size=0.2, random_state=42
# )

# # -------------------------
# # Train Model
# # -------------------------
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)

# # -------------------------
# # Accuracy
# # -------------------------
# y_pred = model.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, y_pred))

# # -------------------------
# # Save model and vectorizer
# # -------------------------
# with open("model.pkl", "wb") as f:
#     pickle.dump(model, f)

# with open("vectorizer.pkl", "wb") as f:
#     pickle.dump(vectorizer, f)

# print("Model saved successfully!")
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -------------------------
# Load datasets
# -------------------------
df_fake = pd.read_csv("news_dataset.csv/Fake.csv")
df_true = pd.read_csv("news_dataset.csv/True.csv")

# -------------------------
# Add labels
# -------------------------
df_fake["label"] = 0   # Fake News
df_true["label"] = 1   # Real News

# -------------------------
# Combine datasets
# -------------------------
df = pd.concat([df_fake, df_true], axis=0)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Class Distribution:")
print(df["label"].value_counts())

# -------------------------
# Combine title + text
# -------------------------
df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")

# -------------------------
# Clean text
# -------------------------
df["content"] = df["content"].str.lower()
df["content"] = df["content"].str.replace("[^a-zA-Z ]", "", regex=True)

# -------------------------
# OPTIONAL: Reduce dataset size for faster training
# (Remove this line if you want full dataset)
# -------------------------
# df = df.sample(15000, random_state=42)

# -------------------------
# Features and labels
# -------------------------
X = df["content"]
y = df["label"]

# -------------------------
# TF-IDF Vectorization (Optimized)
# -------------------------
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000   
)

X_vectorized = vectorizer.fit_transform(X)

# -------------------------
# Train/Test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# -------------------------
# Train Model
# -------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -------------------------
# Accuracy
# -------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# -------------------------
# Save model and vectorizer
# -------------------------
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model saved successfully!")
