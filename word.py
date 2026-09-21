from fastapi import FastAPI
from pydantic import BaseModel

import tensorflow as tf
import pickle
import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences


# Load model
model = tf.keras.models.load_model("lstm_model.keras")


# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)


MAX_LEN = 56


app = FastAPI(
    title="LSTM Text Prediction API",
    description="API for LSTM based text prediction",
    version="1.0"
)


class TextRequest(BaseModel):
    text: str


@app.get("/")
def home():
    return {
        "message": "LSTM API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(request: TextRequest):

    text = request.text

    # Convert text to sequence
    sequence = tokenizer.texts_to_sequences([text])

    # Padding
    padded_sequence = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="pre"
    )

    # Prediction
    prediction = model.predict(
        padded_sequence,
        verbose=0
    )

    # Top 5 predictions
    top_indices = np.argsort(prediction[0])[-5:][::-1]

    results = []

    for idx in top_indices:

        word = tokenizer.index_word.get(
            int(idx),
            "<unknown>"
        )

        confidence = float(prediction[0][idx])

        results.append({
            "word": word,
            "confidence": confidence
        })

    return {
        "input": request.text,
        "predictions": results
    }