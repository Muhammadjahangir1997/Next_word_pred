# import streamlit as st
# import requests

# st.set_page_config(page_title="Next Word Prediction", page_icon="🤖")

# st.title("Next Word Prediction System")
# st.write("Enter text below and click Predict Next Words")

# # --------------------
# # User Input
# # --------------------
# user_text = st.text_input("Enter starting text")
# n_words = st.number_input("Number of words to predict", min_value=1, max_value=20, value=5)

# # --------------------
# # Predict Button
# # --------------------
# if st.button("Predict Next Words"):
#     if user_text.strip() == "":
#         st.warning("Please enter some text")
#     else:
#         try:
#             response = requests.post(
#                 "http://127.0.0.1:8000/predict",
#                 json={"text": user_text, "n_words": n_words}
#             )

#             if response.status_code == 200:
#                 result = response.json()
#                 st.success(f"Predicted Text: {result['predicted_text']}")
#             else:
#                 st.error("API Error: Unable to get prediction")

#         except Exception as e:
#             st.error(f"Connection Error: {e}")
import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.title("Next Word Prediction")

# --------------------
# Load model & tokenizer (once)
# --------------------
model = load_model("next_word_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tok = pickle.load(f)

# --------------------
# Prediction function
# --------------------
def predict_text(text, n_words=5):
    text = text.lower()
    for _ in range(n_words):
        seq = tok.texts_to_sequences([text])[0]
        seq = pad_sequences([seq], maxlen=20)
        pred = model.predict(seq, verbose=0)
        next_index = np.argmax(pred)
        next_word = ""
        for word, index in tok.word_index.items():
            if index == next_index:
                next_word = word
                break
        if next_word == "":
            break
        text += " " + next_word
    return text

# --------------------
# Streamlit UI
# --------------------
user_text = st.text_input("Enter starting text")
n_words = st.number_input("Number of words to predict", min_value=1, max_value=20, value=5)

if st.button("Predict"):
    if user_text.strip() == "":
        st.warning("Please enter some text")
    else:
        result = predict_text(user_text, n_words)
        st.success(f"Predicted Text: {result}")
