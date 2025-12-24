import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Misol SMS xabarlari (ham va spam)
data = [
    ("spam", "Win a FREE iPhone now! Click here!"),
    ("spam", "Congratulations! You won $1000 cash!"),
    ("ham", "Salom, kechqurun uchrashamizmi?"),
    ("ham", "Bugun dars soat 5da boshlanadi."),
]

df = pd.DataFrame(data, columns=["label", "text"])

# Targetni 0/1 ga aylantirish
df["target"] = df["label"].apply(lambda x: 1 if x == "spam" else 0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["target"], test_size=0.2, random_state=42
)

# TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

# LightGBM classifier
model = LGBMClassifier()
model.fit(X_train_vec, y_train)

# Fayllarni pickle formatida saqlash
with open("spam_model_lgb.pkl", "wb") as f:
    pickle.dump(model, f)

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model va vectorizer muvaffaqiyatli saqlandi!")
