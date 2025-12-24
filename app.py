import streamlit as st
import pickle

# Model va vectorizerni yuklash funksiyasi
@st.cache_data
def load_model():
    try:
        with open("spam_model_lgb.pkl", "rb") as f:
            model = pickle.load(f)
        with open("tfidf_vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except Exception as e:
        st.error(f"Model yoki vectorizer yuklashda xatolik: {e}")
        return None, None

def main():
    st.title("📨 SMS va Email Spam Detector")

    # Model va vectorizerni yuklash
    model, vectorizer = load_model()
    if model is None or vectorizer is None:
        st.stop()  # Agar yuklashda xatolik bo'lsa, dastur to'xtaydi

    # Foydalanuvchi xabarini kiritish
    text = st.text_area("Xabarni kiriting:")

    if st.button("Tekshirish"):
        if not text.strip():
            st.warning("Iltimos, xabarni kiriting!")
            return

        # Xabarni vectorizer orqali transformatsiya qilish
        vec = vectorizer.transform([text])

        # Bashorat qilish
        pred = model.predict(vec)[0]
        prob_spam = model.predict_proba(vec)[0][1]  # Spam ehtimoli

        # Natijani ko'rsatish
        if pred == 1:
            st.error(f"🚨 SPAM aniqlandi! (Ehtimol: {prob_spam:.1%})")
            st.info("Ehtiyot bo'ling – bu reklama yoki firibgarlik bo'lishi mumkin.")
        else:
            st.success(f"✅ Haqiqiy xabar (HAM) (Spam ehtimoli: {prob_spam:.1%})")

        st.write("**Sizning xabaringiz:**")
        st.code(text)

if __name__ == "__main__":
    main()
