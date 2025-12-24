import streamlit as st
import pickle

@st.cache_data
def load_model():
    with open("spam_model_lgb.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

def main():
    st.title("SMS va Email Spam Detector")

    try:
        model, vectorizer = load_model()
    except Exception as e:
        st.error(f"Model yoki vectorizer yuklashda xatolik: {e}")
        return

    text = st.text_area("Xabarni kiriting:")

    if st.button("Tekshirish"):
        if not text.strip():
            st.warning("Iltimos, xabarni kiriting!")
            return

        vec = vectorizer.transform([text])
        pred = model.predict(vec)[0]
        prob_spam = model.predict_proba(vec)[0][1]

        if pred == 1:
            st.error(f"🚨 SPAM aniqlandi! (Ehtimol: {prob_spam:.1%})")
            st.info("Ehtiyot bo'ling – bu reklama yoki firibgarlik bo'lishi mumkin.")
        else:
            st.success(f"✅ Haqiqiy xabar (HAM) (Spam ehtimoli: {prob_spam:.1%})")

        st.write("**Sizning xabaringiz:**")
        st.code(text)

if __name__ == "__main__":
    main()
