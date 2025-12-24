import streamlit as st
import pickle

# ===============================
# STREAMLIT CONFIG
# ===============================
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
    layout="centered"
)

st.title("📩 SMS / Email Spam Detector")

# ===============================
# MODEL & VECTORIZER LOAD
# ===============================
@st.cache_resource
def load_model():
    with open("spam_model_lgb.pkl", "rb") as f:
        model = pickle.load(f)

    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer

model, vectorizer = load_model()

st.success("✅ Model va vectorizer muvaffaqiyatli yuklandi")

# ===============================
# INPUT UI
# ===============================
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

# ===============================
# PREDICTION
# ===============================
if st.button("🔍 Tekshirish"):
    if text.strip() == "":
        st.warning("⚠️ Iltimos, xabar matnini kiriting.")
    else:
        vec = vectorizer.transform([text])

        # HAR DOIM spam klassi = index 1
        spam_prob = model.predict_proba(vec)[0][1]

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
st.caption("🚀 LightGBM + TF-IDF | Pretrained Model")
