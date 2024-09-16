import numpy as np 
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence


# load the imdb word index

word_index = imdb.get_word_index()

reversed_word_index = {value: key for key, value in word_index.items()}

# load the model

model = load_model('simple_rnn_imdb2.h5')


# helper functions (putting here instead of importing)

def dec_review(encoded_review):
    return ' '.join([reversed_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen= 500)
    return padded_review


# prediction function

def predict_sentiment(review):
    
    padded_text = preprocess_text(review)
    
    prediction = model.predict(padded_text)
    
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    
    return sentiment, prediction[0][0]



#################### streamlit app design ##################


import streamlit as st

st.title("IMDB Review Sentiment Analysis")
st.write("Enter a movie review to see if its positive or negative")


# user input
input = st.text_area("movie review")

# create a button for classification

if st.button("submit"):
    
    preprocessed_input = preprocess_text(input)
    
    prediction = model.predict(preprocessed_input)
    
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    
    # desplay the result
    
    st.write(f"sentiment is {sentiment}")
    st.write(f"prediction score is {prediction[0][0]}")
    
else:
    st.write('Enter the movie review')

    
    


