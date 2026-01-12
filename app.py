if st.button("Tekshirish"):
            if text.strip():
                # Bashorat qilish
                vec = vectorizer.transform([text])
                prediction = model.predict(vec)[0]
                probability = model.predict_proba(vec)[0][1]

                st.divider()
                
                # Vizual natija qismi
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    if prediction == 1:
                        st.error("🚨 Natija: SPAM")
                    else:
                        st.success("✅ Natija: HAQIQIY (HAM)")
                
                with col2:
                    st.metric(label="Spam bo'lish ehtimoli", value=f"{probability:.1%}")

                # Progress bar orqali ko'rsatish
                st.write("**Ishonch darajasi:**")
                if probability > 0.5:
                    st.progress(probability, "Spam xavfi yuqori!")
                else:
                    st.progress(probability, "Xabar xavfsiz ko'rinmoqda.")
                
                st.info(f"Tahlil: Model ushbu xabarni {1-probability if prediction == 0 else probability:.1%} ishonch bilan klassifikatsiya qildi.")

            else:
                st.warning("Iltimos, matn kiriting!")
