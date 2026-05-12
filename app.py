# app.py

import streamlit as st
import pickle

# Load model & vectorizer
model = pickle.load(open("model/spam_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

st.set_page_config(page_title="Spam Detector", page_icon="📧")

st.title("📧 Spam Email Detection System")
st.write("Enter a message to check whether it's Spam or Not")

# Input
msg = st.text_area("Enter your message here:")

# Button
if st.button("Predict"):
    if msg.strip() == "":
        st.warning("Please enter a message")
    else:
        msg_vec = vectorizer.transform([msg])
        prediction = model.predict(msg_vec)[0]

        if prediction == 1:
            st.error("🚫 SPAM MESSAGE")
        else:
            st.success("✅ NOT SPAM")