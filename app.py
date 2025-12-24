pip install streamlit scikit-learn lightgbm pandas

import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

# ===============================
# 1. DATASETNI YUKLASH
# ===============================
# SMS Spam Collection format: label \t text
df = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "text"]
)

# Label encoding
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# ===============================
# 2. TRAIN / TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# ===============================
# 3. VECTORIZER
# ===============================
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=5000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ===============================
# 4. LIGHTGBM MODEL
# ===============================
model = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.1,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_vec, y_train)

# ===============================
# 5. MODEL TEKSHIRISH
# ===============================
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)

# ===============================
# 6. STREAMLIT UI
# ===============================
st.set_page_config(page_title="SMS Spam Detector", page_icon="📩")

st.title("📩 SMS / Email Spam Detector")
st.write(f"📊 Model aniqligi (Accuracy): **{acc:.2%}**")

text = st.text_area(
    "Xabar matnini kiriting:",
    height=150,
    placeholder="Masalan: Congratulations! You won a free prize..."
)

if st.button("Tekshirish"):
    if text.strip() == "":
        st.warning("⚠️ Iltimos, xabar matnini kiriting.")
    else:
        # Vectorize input
        vec = vectorizer.transform([text])

        # Prediction
        pred = model.predict(vec)[0]   # 0 = HAM, 1 = SPAM
        spam_prob = model.predict_proba(vec)[0][1]  # HAR DOIM spam ehtimoli

        # Threshold (xohlasang o‘zgartir)
        threshold = 0.4

        if spam_prob >= threshold:
            st.error(f"🚨 SPAM aniqlandi! (Spam ehtimoli: {spam_prob:.1%})")
            st.info("Ehtiyot bo‘ling – bu reklama yoki firibgarlik bo‘lishi mumkin.")
        else:
            st.success(f"✅ Haqiqiy xabar (HAM) (Spam ehtimoli: {spam_prob:.1%})")

        st.write("**Sizning xabaringiz:**")
        st.code(text)
