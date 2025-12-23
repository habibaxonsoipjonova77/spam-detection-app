import streamlit as st
import joblib

# Model va vectorizerni yuklash
model = joblib.load('spam_model_lgb.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

st.title("📱 SMS va Email Spam Detector")
st.write("Xabarni kiriting – spam yoki haqiqiy ekanligini aniqlayman!")
st.write("Model: LightGBM | Dataset: SMS Spam Collection")

text = st.text_area("Xabar matni:", height=150, placeholder="Masalan: Free money win prize...")

if st.button("Tekshirish", type="primary"):
    if not text.strip():
        st.warning("Iltimos, xabar kiriting!")
    else:
        vec = vectorizer.transform([text])
        pred = model.predict(vec)[0]
        prob_spam = model.predict_proba(vec)[0][1]  # Har doim spam klassining ehtimoli (index 1)

        if pred == 1:
            st.error(f"🚨 SPAM aniqlandi! (Ehtimol: {prob_spam:.1%})")
            st.info("Ehtiyot bo'ling – bu reklama yoki firibgarlik bo'lishi mumkin.")
        else:
            st.success(f"✅ Haqiqiy xabar (HAM) (Spam ehtimoli: {prob_spam:.1%})")
        
        st.write("**Sizning xabaringiz:**")
        st.code(text)
