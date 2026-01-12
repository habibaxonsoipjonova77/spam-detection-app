import streamlit as st
import joblib
import os

# Sahifa sarlavhasi
st.set_page_config(page_title="Spam Detector", page_icon="📨")

@st.cache_resource
def load_assets():
    try:
        # Fayllar mavjudligini tekshirish
        if not os.path.exists("spam_model_lgb.pkl") or not os.path.exists("tfidf_vectorizer.pkl"):
            return None, None
        
        model = joblib.load("spam_model_lgb.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        return model, vectorizer
    except Exception as e:
        st.error(f"Yuklashda xatolik: {e}")
        return None, None

def main():
    st.title("📨 SMS va Email Spam Detector")
    st.markdown("Xabaringizni pastga kiriting va biz uni spam yoki yo'qligini aniqlaymiz.")

    model, vectorizer = load_assets()

    if model is None:
        st.error("❌ Model fayllari topilmadi! Iltimos, .pkl fayllari GitHub'ga to'g'ri yuklanganiga ishonch hosil qiling.")
        return

    text = st.text_area("Xabarni kiriting:", height=150)

    if st.button("Tekshirish"):
        if text.strip():
            # Bashorat qilish
            vec = vectorizer.transform([text])
            prediction = model.predict(vec)[0]
            probability = model.predict_proba(vec)[0][1]

            st.divider()
            if prediction == 1:
                st.error(f"🚨 **SPAM aniqlandi!** (Ehtimollik: {probability:.1%})")
            else:
                st.success(f"✅ **Bu haqiqiy xabar (HAM)** (Spam ehtimoli: {probability:.1%})")
        else:
            st.warning("Iltimos, matn kiriting!")

if __name__ == "__main__":
    main()
