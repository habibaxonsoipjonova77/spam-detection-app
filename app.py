import streamlit as st
import pickle
import numpy as np

# Model va vectorizerni yuklash funksiyasi
@st.cache_data(show_spinner=True)
def load_model():
    with open("spam_model_lgb.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

def main():
    st.title("SMS va Email Spam Detector")

    model, vectorizer = load_model()

    text = st.text_area("Xabarni kiriting:")

    if st.button("Tekshirish"):
        if not text.strip():
            st.warning("Iltimos, xabarni kiriting!")
            return
        
        vec = vectorizer.transform([text])
        pred = model.predict(vec)[0]
        prob_spam = model.predict_proba(vec)[0][1]  # Spam ehtimoli

        if pred == 1:
            st.error(f"🚨 SPAM aniqlandi! (Ehtimol: {prob_spam:.1%})")
            st.info("Ehtiyot bo'ling – bu reklama yoki firibgarlik bo'lishi mumkin.")
        else:
            st.success(f"✅ Haqiqiy xabar (HAM) (Spam ehtimoli: {prob_spam:.1%})")

        st.write("**Sizning xabaringiz:**")
        st.code(text)

if __name__ == "__main__":
    main()
