import streamlit as st
import pickle
import re

# Load model & vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text)
    return text

# Streamlit UI
st.title("📰 Fake News Detection System")
st.write("Enter a news headline or article text to check if it is **Fake** or **Real**.")

user_input = st.text_area("Enter News Text Here...")

if st.button("Check"):
    cleaned = clean_text(user_input)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]

    if prediction == 0:
        st.error(f"🚨 Fake News Detected! (Confidence: {prob[0]*100:.2f}%)")
    else:
        st.success(f"✅ Real News Detected! (Confidence: {prob[1]*100:.2f}%)")