import tensorflow as tf
import numpy as np

# Download the Shakespeare text dataset
path = tf.keras.utils.get_file("shakespeare.txt",
                               "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt")
text = open(path, 'rb').read().decode(encoding='utf-8')
print(f"Length of text: {len(text)} characters")

# Create a vocabulary of unique characters and mappings
vocab = sorted(set(text))
print(f"{len(vocab)} unique characters")

char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

# Convert the text into integers
text_as_int = np.array([char2idx[c] for c in text])

# Set the sequence length for training examples
seq_length = 100
examples_per_epoch = len(text) // (seq_length + 1)

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

# Create training batches
BATCH_SIZE = 64
BUFFER_SIZE = 10000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# Build the RNN model
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[BATCH_SIZE, None]),
    tf.keras.layers.LSTM(rnn_units,
                         return_sequences=True,
                         stateful=True,
                         recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
])

# Define the loss function
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

model.compile(optimizer='adam', loss=loss)

# Train the model for 1 epoch (for demonstration; use more epochs for better results)
EPOCHS = 1
history = model.fit(dataset, epochs=EPOCHS)

# For text generation, rebuild the model with batch size 1 and load the trained weights.
model_for_generation = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[1, None]),
    tf.keras.layers.LSTM(rnn_units,
                         return_sequences=True,
                         stateful=True,
                         recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
])
model_for_generation.set_weights(model.get_weights())

def generate_text(model, start_string, num_generate=500):
    # Convert the start string to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    
    # Empty list to store generated characters
    text_generated = []
    
    # Temperature parameter affects randomness in predictions.
    temperature = 1.0
    
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        
        # Adjust predictions by the temperature
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        
        # Pass the predicted character as the next input to the model
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])
    
    return start_string + ''.join(text_generated)

# Generate and print sample text starting with "ROMEO: "
print("\nGenerated Text:\n")
print(generate_text(model_for_generation, start_string="ROMEO: "))
