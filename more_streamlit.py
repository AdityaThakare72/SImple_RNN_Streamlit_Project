import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
import streamlit as st

# Load the IMDb word index
word_index = imdb.get_word_index()
reversed_word_index = {value: key for key, value in word_index.items()}

# Load the model
model = load_model('simple_rnn_imdb2.h5')

# Helper functions
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

def predict_sentiment(review):
    padded_text = preprocess_text(review)
    prediction = model.predict(padded_text)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]

# Streamlit app design
st.set_page_config(page_title="IMDB Review Sentiment Analysis", page_icon=":movie_camera:", layout="wide")

st.title("🎬 IMDB Review Sentiment Analysis")
st.subheader("Check the sentiment of movie reviews!")

st.markdown("""
    **Instructions:**
    - Enter a movie review in the text box below.
    - Click the **Submit** button to get the sentiment analysis.
    - The model will classify the review as **Positive** or **Negative** based on the content.
    - The prediction score indicates the confidence of the sentiment.
""")

# Sidebar for additional information or options
st.sidebar.header("About")
st.sidebar.info("""
    This application uses a Simple RNN model to analyze the sentiment of movie reviews from the IMDB dataset.
    - **Model**: Simple RNN
    - **Dataset**: IMDB Reviews
    - **Usage**: Enter a review and get sentiment analysis.
""")

# User input
input_review = st.text_area("Movie Review", placeholder="Type your review here...")

# Create a button for classification
if st.button("Submit"):
    if input_review.strip():
        try:
            sentiment, score = predict_sentiment(input_review)
            st.markdown(f"### **Sentiment Analysis Result**")
            st.write(f"**Sentiment**: {sentiment}")
            st.write(f"**Prediction Score**: {score:.2f}")
            st.markdown("**Feedback**: Based on the score, the review is classified as positive or negative.")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.warning("Please enter a valid movie review.")
else:
    st.write("Enter the movie review and click submit to see the results.")

