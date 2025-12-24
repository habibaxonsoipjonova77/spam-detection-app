import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

# ===============================
# STREAMLIT CONFIG
# ===============================
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
    layout="centered"
)

# ===============================
# DATASET LOAD
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv(
        "SMSSpamCollection",
        sep="\t",
        header=None,
        names=["label", "text"]
    )
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    return df

df = load_data()

# ===============================
# TRAIN / TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# ===============================
# VECTORIZER
# ===============================
@st.cache_resource
def train_vectorizer_and_model(X_train, y_train):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=5000
    )

    X_train_vec = vectorizer.fit_transform(X_train)

    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.1,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train_vec, y_train)
    return vectorizer, model

vectorizer, model = train_vectorizer_and_model(X_train, y_train)

# ===============================
# MODEL ACCURACY
# ===============================
X_test_vec = vectorizer.transform(X_test)
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

# ===============================
# UI
# ===============================
st.title("📩 SMS / Email Spam Detector")
st.write(f"📊 Model aniqligi (Accuracy): **{accuracy:.2%}**")

st.markdown("---")

text = st.text_area(
    "Xabar matnini kiriting:",
    height=150,
    placeholder="Masalan: Congratulations! You won a free prize..."
)

threshold = st.slider(
    "Spam threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.4,
    step=0.05
)

if st.button("🔍 Tekshirish"):
    if text.strip() == "":
        st.warning("⚠️ Iltimos, xabar matnini kiriting.")
    else:
        vec = vectorizer.transform([text])

        # Prediction
        spam_prob = model.predict_proba(vec)[0][1]  # HAR DOIM spam = index 1

        if spam_prob >= threshold:
            st.error(f"🚨 SPAM aniqlandi! (Spam ehtimoli: {spam_prob:.1%})")
            st.info("Ehtiyot bo‘ling – bu reklama yoki firibgarlik bo‘lishi mumkin.")
        else:
            st.success(f"✅ Haqiqiy xabar (HAM) (Spam ehtimoli: {spam_prob:.1%})")

        st.write("**Sizning xabaringiz:**")
        st.code(text)

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.caption("🚀 LightGBM + SMS Spam Collection | Streamlit App")
