import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Load your dataset
df = pd.read_csv('data.csv')

# Split the dataset into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(df['sentence'], df['spam'], test_size=0.2, random_state=42)

# Tokenize the text
max_words = 10000  # Adjust as needed
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_texts)

# Convert text to sequences
train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

# Pad sequences to a fixed length
max_sequence_length = 100  # Adjust as needed
train_data = pad_sequences(train_sequences, maxlen=max_sequence_length)
test_data = pad_sequences(test_sequences, maxlen=max_sequence_length)

# Build the RCNN model
model_rcnn = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=max_words, output_dim=32, input_length=max_sequence_length),
    tf.keras.layers.Conv1D(64, 5, activation='relu'),
    tf.keras.layers.SimpleRNN(64, activation='relu', return_sequences=True),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model_rcnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model_rcnn.fit(train_data, train_labels, epochs=5, batch_size=32, validation_split=0.2)

# Evaluate the model
loss_rcnn, accuracy_rcnn = model_rcnn.evaluate(test_data, test_labels)
print(f'Test accuracy (RCNN): {accuracy_rcnn}')
