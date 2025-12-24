vec = vectorizer.transform([text])

# Prediction
pred = model.predict(vec)[0]  # 0 = HAM, 1 = SPAM

# Probability (HAR DOIM spam klassi = index 1)
spam_prob = model.predict_proba(vec)[0][1]

# Natija chiqarish
if pred == 1:
    st.error(f"🚨 SPAM aniqlandi! (Spam ehtimoli: {spam_prob:.1%})")
    st.info("Ehtiyot bo'ling – bu reklama yoki firibgarlik bo'lishi mumkin.")
else:
    st.success(f"✅ Haqiqiy xabar (HAM) (Spam ehtimoli: {spam_prob:.1%})")

st.write("**Sizning xabaringiz:**")
st.code(text)
