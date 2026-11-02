import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense

# Sample text data
texts = [
    "Deep learning models learn language",
    "Word embeddings map words to vectors",
    "Neural networks use embeddings for NLP"
]

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index

# Padding sequences
data = pad_sequences(sequences, padding='post')

# Labels
labels = np.array([0, 1, 0])

# Build the model
model = Sequential([
    Embedding(
        input_dim=len(word_index) + 1,
        output_dim=8
    ),
    Flatten(),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy'
)

# Train the model
model.fit(data, labels, epochs=10, verbose=0)

# Extract word embeddings
embeddings = model.layers[0].get_weights()[0]

print("Vocabulary:", word_index)
print(
    "Embedding for 'embeddings':",
    embeddings[word_index['embeddings']]
)