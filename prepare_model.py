import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

# 1. Ma'lumotlarni tayyorlash
data = [
    ("spam", "Win a FREE iPhone now! Click here!"),
    ("spam", "Congratulations! You won $1000 cash!"),
    ("spam", "Urgent: Your account has been compromised. Update now."),
    ("ham", "Salom, kechqurun uchrashamizmi?"),
    ("ham", "Bugun dars soat 5da boshlanadi."),
    ("ham", "Ertaga futbol o'ynaymizmi? Kelasanmi?"),
]

df = pd.DataFrame(data, columns=["label", "text"])
df["target"] = df["label"].apply(lambda x: 1 if x == "spam" else 0)

# 2. Vectorizer va Modelni o'qitish
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["target"]

model = LGBMClassifier()
model.fit(X, y)

# 3. Fayllarni saqlash (Joblib ishonchliroq)
joblib.dump(model, "spam_model_lgb.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("✅ Model va vectorizer muvaffaqiyatli saqlandi!")
