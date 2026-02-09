Next Word Prediction System

This project implements a Next Word Prediction system using NLP.

Backend: FastAPI
Frontend: Streamlit
Model: LSTM (TensorFlow / Keras)

The trained model is saved as .h5 and tokenizer as .pkl.
Streamlit sends user input to FastAPI, which predicts the next words
and returns the result.

Run Instructions:
1. uvicorn app:app --reload
2. streamlit run app.py
