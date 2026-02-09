# =========================
# IMPORTS
# =========================
import pickle
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =========================
# LOAD MODEL & TOKENIZER
# =========================
model = load_model("next_word_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tok = pickle.load(f)

# =========================
# FASTAPI APP
# =========================
app = FastAPI(title="Next Word Prediction API")

# =========================
# REQUEST SCHEMA
# =========================
class TextInput(BaseModel):
    text: str
    n_words: int = 5   # kitne next words chahiye

# =========================
# PREDICTION ENDPOINT
# =========================
@app.post("/predict")
def predict_next_words(data: TextInput):

    text = data.text.lower()

    for _ in range(data.n_words):

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

        text = text + " " + next_word

    return {
        "input_text": data.text,
        "predicted_text": text
    }

# =========================
# ROOT ENDPOINT (OPTIONAL)
# =========================
@app.get("/")
def home():
    return {"message": "Next Word Prediction API is running"}
