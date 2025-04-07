# frontend/app.py

import streamlit as st
import requests

st.set_page_config(page_title="Sentiment Analysis", page_icon="💬")
st.title("💬 Sentiment Analysis App")

text_input = st.text_area("Enter a movie review:")

if st.button("Analyze Sentiment"):
    if text_input.strip():
        with st.spinner("Analyzing..."):
            res = requests.post("http://127.0.0.1:5000/predict", json={"text": text_input})
            if res.status_code == 200:
                sentiment = res.json()['sentiment']
                st.success(f"Sentiment: {sentiment}")
            else:
                st.error("Error from API.")
    else:
        st.warning("Please enter text first.")