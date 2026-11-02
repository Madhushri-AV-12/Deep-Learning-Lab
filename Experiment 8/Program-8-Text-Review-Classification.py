
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Set random seed
tf.random.set_seed(42)
np.random.seed(42)


class TextReviewClassifier:

    def __init__(self, vocab_size=10000, max_length=200, embedding_dim=100):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.tokenizer = None
        self.model = None
        self.history = None

        self.stop_words = set(stopwords.words('english'))
        # Keep negation words because they affect sentiment
        self.stop_words -= {'not', 'no', 'nor'}
        self.lemmatizer = WordNetLemmatizer()

    # 1. Preprocess text
    def preprocess_text(self, text):

        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        words = text.split()

        words = [
            word for word in words
            if word not in self.stop_words
        ]

        words = [
            self.lemmatizer.lemmatize(word)
            for word in words
        ]

        return ' '.join(words)

    # 2. Load and preprocess dataset
    def load_and_preprocess_data(self, file_path=None):

        if file_path:
            df = pd.read_csv(file_path)
            texts = df['text'].astype(str).values
            labels = df['sentiment'].values

        else:
            print("Loading IMDB dataset...")

            (x_train, y_train), (x_test, y_test) = (
                tf.keras.datasets.imdb.load_data(
                    num_words=self.vocab_size
                )
            )

            word_index = tf.keras.datasets.imdb.get_word_index()

            reverse_word_index = {
                value: key for key, value in word_index.items()
            }

            def decode_review(encoded_review):
                return " ".join([
                    reverse_word_index.get(i - 3, "?")
                    for i in encoded_review
                    if i >= 3
                ])

            x = np.concatenate([x_train, x_test])
            y = np.concatenate([y_train, y_test])

            # Use 5000 reviews for faster execution
            texts = [
                decode_review(review)
                for review in x[:5000]
            ]

            labels = y[:5000]

        print("Preprocessing texts...")

        texts = [
            self.preprocess_text(text)
            for text in texts
        ]

        return texts, np.asarray(labels, dtype=np.int32)

    # 3. Tokenization and padding
    def prepare_data(self, texts, labels):

        self.tokenizer = Tokenizer(
            num_words=self.vocab_size,
            oov_token='<OOV>'
        )

        self.tokenizer.fit_on_texts(texts)

        sequences = self.tokenizer.texts_to_sequences(texts)

        padded_sequences = pad_sequences(
            sequences,
            maxlen=self.max_length,
            padding='post',
            truncating='post'
        )

        # Separate training and testing data
        X_train, X_test, y_train, y_test = train_test_split(
            padded_sequences,
            labels,
            test_size=0.2,
            random_state=42,
            stratify=labels
        )

        # Separate validation data from training data
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=42,
            stratify=y_train
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    # 4. Build deep learning model
    def build_model(self):

        self.model = Sequential([
            Embedding(
                self.vocab_size,
                self.embedding_dim
            ),

            Bidirectional(
                LSTM(64, return_sequences=True)
            ),

            Dropout(0.5),

            Bidirectional(LSTM(32)),

            Dense(64, activation='relu'),

            Dropout(0.5),

            Dense(1, activation='sigmoid')
        ])

        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return self.model

    # 5. Train model
    def train(self, X_train, y_train, X_val, y_val,
              epochs=5, batch_size=32):

        self.history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            verbose=1
        )

        return self.history

    # 6. Evaluate model
    def evaluate(self, X_test, y_test):

        loss, accuracy = self.model.evaluate(
            X_test, y_test, verbose=0
        )

        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy * 100:.2f}%")

        predictions = self.model.predict(
            X_test, verbose=0
        )

        predicted_classes = (
            predictions > 0.5
        ).astype(int).flatten()

        print("\nClassification Report:")

        print(classification_report(
            y_test,
            predicted_classes,
            target_names=['Negative', 'Positive'],
            zero_division=0
        ))

        # Confusion matrix
        cm = confusion_matrix(
            y_test,
            predicted_classes
        )

        plt.figure(figsize=(6, 4))

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive']
        )

        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

        return loss, accuracy, predictions

    # 7. Predict sentiment of new text
    def predict_text(self, text):

        text = self.preprocess_text(text)

        sequence = self.tokenizer.texts_to_sequences([text])

        padded = pad_sequences(
            sequence,
            maxlen=self.max_length,
            padding='post',
            truncating='post'
        )

        prediction = self.model.predict(
            padded, verbose=0
        )

        score = prediction[0][0]

        sentiment = (
            "Positive" if score > 0.5 else "Negative"
        )

        confidence = (
            score if score > 0.5 else 1 - score
        )

        return sentiment, confidence

    # 8. Plot training history
    def plot_training_history(self):

        if self.history is None:
            print("No training history available")
            return

        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(12, 4)
        )

        ax1.plot(
            self.history.history['accuracy'],
            label='Training Accuracy'
        )

        ax1.plot(
            self.history.history['val_accuracy'],
            label='Validation Accuracy'
        )

        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()

        ax2.plot(
            self.history.history['loss'],
            label='Training Loss'
        )

        ax2.plot(
            self.history.history['val_loss'],
            label='Validation Loss'
        )

        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()

        plt.tight_layout()
        plt.show()


# 9. Main program
if __name__ == "__main__":

    print("Initializing Text Review Classifier...")

    classifier = TextReviewClassifier(
        vocab_size=10000,
        max_length=200,
        embedding_dim=100
    )

    # Load dataset
    texts, labels = classifier.load_and_preprocess_data()

    # Prepare dataset
    X_train, X_val, X_test, y_train, y_val, y_test = (
        classifier.prepare_data(texts, labels)
    )

    print("Training data shape:", X_train.shape)
    print("Validation data shape:", X_val.shape)
    print("Testing data shape:", X_test.shape)

    # Build model
    print("\nBuilding model...")

    model = classifier.build_model()
    model.summary()

    # Train model
    print("\nTraining model...")

    classifier.train(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=5,
        batch_size=32
    )

    # Evaluate model
    print("\nEvaluating model...")

    loss, accuracy, predictions = classifier.evaluate(
        X_test, y_test
    )

    # Plot training history
    classifier.plot_training_history()

    # Save model
    model.save("sentiment_analysis_model.h5")

    print("\nModel saved successfully!")

    # Sample predictions
    sample_reviews = [
        "This movie was absolutely fantastic! I loved every minute of it.",
        "The acting was terrible and the plot was boring. I hated it.",
        "It was okay, not great but not terrible either.",
        "This product is amazing and works perfectly. Highly recommend!",
        "Poor quality and waste of money. Would not buy again."
    ]

    print("\nSample Predictions:")

    for review in sample_reviews:

        sentiment, confidence = classifier.predict_text(
            review
        )

        print("\nReview:", review)
        print("Sentiment:", sentiment)
        print(f"Confidence: {confidence:.2f}")

    print("\nText Review Classification Completed Successfully!")
