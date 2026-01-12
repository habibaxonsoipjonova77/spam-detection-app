import streamlit as st
import joblib
import os
from transformers import pipeline

# Sahifa sarlavhasi
st.set_page_config(page_title="AI Spam Detector", page_icon="🤖", layout="wide")

@st.cache_resource
def load_assets():
    try:
        # 1. Klassik modelni yuklash
        model = joblib.load("spam_model_lgb.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        
        # 2. Zamonaviy Deep Learning (AI) modelini yuklash
        # Bu model Hugging Face-dan avtomatik yuklanadi
        ai_model = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-sms-spam-detection")
        
        return model, vectorizer, ai_model
    except Exception as e:
        st.error(f"Modellarni yuklashda xatolik: {e}")
        return None, None, None

def main():
    st.title("📨 SMS va Email Spam Detector (Classic + AI)")
    
    model, vectorizer, ai_model = load_assets()

    if model is None or ai_model is None:
        st.error("Modellarni yuklab bo'lmadi. Fayllar mavjudligini tekshiring.")
        return

    text = st.text_area("Xabarni kiriting:", height=150, placeholder="Masalan: Congratulations! You won a prize...")

    if st.button("Tahlil qilish"):
        if text.strip():
            st.divider()
            
            # Ikki xil tahlil uchun ustunlar
            col1, col2 = st.columns(2)
            
            # --- 1-qism: Klassik Machine Learning Natijasi ---
            with col1:
                st.subheader("📊 Klassik Model (ML)")
                vec = vectorizer.transform([text])
                prediction = model.predict(vec)[0]
                prob = model.predict_proba(vec)[0][1]
                
                if prediction == 1:
                    st.error(f"SPAM! (Ehtimol: {prob:.1%})")
                else:
                    st.success(f"HAM (Haqiqiy) (Ehtimol: {prob:.1%})")
                st.caption("Ushbu natija TF-IDF va LightGBM orqali hisoblandi.")

            # --- 2-qism: Zamonaviy Deep Learning (BERT) Natijasi ---
            with col2:
                st.subheader("🧠 Sun'iy Intellekt (AI)")
                ai_result = ai_model(text)[0] # AI tahlili
                label = ai_result['label']
                score = ai_result['score']
                
                # Model natijasini chiroyli ko'rsatish
                if label == "LABEL_1" or label == "spam":
                    st.error(f"SPAM! (Ishonch: {score:.1%})")
                else:
                    st.success(f"HAM (Haqiqiy) (Ishonch: {score:.1%})")
                st.caption("Ushbu natija Transformer (BERT) modeli orqali hisoblandi.")

            # Umumiy xulosa
            st.info("💡 **Tavsiya:** Agar ikki model natijasi farq qilsa, Sun'iy Intellekt (BERT) natijasiga ko'proq ishonish tavsiya etiladi, chunki u gapning ma'nosini tushunadi.")
            
        else:
            st.warning("Iltimos, matn kiriting!")

if __name__ == "__main__":
    main()
