
import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Dense, Dropout, Bidirectional
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

# 1. Load IMDB dataset
print("Loading IMDB dataset...")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
    num_words=10000
)

# 2. Get word index and reverse it
word_index = tf.keras.datasets.imdb.get_word_index()

reverse_word_index = {
    value: key for key, value in word_index.items()
}

# Function to decode reviews
def decode_review(encoded_review):
    return " ".join([
        reverse_word_index.get(i - 3, "?")
        for i in encoded_review
    ])

# Use first 2000 reviews for faster training
texts = [
    decode_review(review) for review in x_train[:2000]
]

labels = y_train[:2000]

# 3. Define parameters
VOCAB_SIZE = 10000
MAX_LENGTH = 200
EMBEDDING_DIM = 100
OOV_TOKEN = '<OOV>'

# 4. Text preprocessing
tokenizer = Tokenizer(
    num_words=VOCAB_SIZE,
    oov_token=OOV_TOKEN
)

tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)

padded_sequences = pad_sequences(
    sequences,
    maxlen=MAX_LENGTH,
    padding='post',
    truncating='post'
)

# Convert to NumPy arrays
X = np.array(padded_sequences)
y = np.array(labels)

# 5. Split dataset
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 6. Build Bidirectional LSTM model
model = Sequential([
    Input(shape=(MAX_LENGTH,)),
    Embedding(VOCAB_SIZE, EMBEDDING_DIM),
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(32)),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# 7. Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Display model architecture
model.summary()

# 8. Train model
print("Training model...")

history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1
)

# 9. Evaluate model
loss, accuracy = model.evaluate(X_val, y_val)

print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# 10. Prediction function
def predict_text(text):
    sequence = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding='post',
        truncating='post'
    )

    prediction = model.predict(padded, verbose=0)

    return prediction[0][0]

# 11. Positive text prediction
sample_text = "This movie was fantastic and I really enjoyed it!"

positive_score = predict_text(sample_text)

print(
    f"Prediction score for positive text: {positive_score:.4f}"
)

# 12. Negative text prediction
sample_text = "This movie was terrible and I hated every minute of it."

negative_score = predict_text(sample_text)

print(
    f"Prediction score for negative text: {negative_score:.4f}"
)
