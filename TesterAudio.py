import speech_recognition as sr
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# Load tokenizer
with open('tokenizer.json', 'r') as f:
    tokenizer = tokenizer_from_json(f.read())

# Parameter untuk model
vocab_size = 10000
embedding_dim = 128
max_length = 128
filters = 64
kernel_size = 3
dense_dim = 16

# Mendefinisikan arsitektur model CNN
def create_cnn_model(vocab_size, embedding_dim, max_length, filters, kernel_size, dense_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Conv1D(filters, kernel_size, activation='relu'),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(dense_dim, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# Mendefinisikan arsitektur model MLP
def create_mlp_model(embedding_dim, dense_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(embedding_dim, activation='relu', input_dim=128),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# Mendefinisikan arsitektur model LSTM
def create_lstm_model(vocab_size, embedding_dim, max_length, filters, dense_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(filters, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(dense_dim, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# Load model weights
def load_model_weights(model, model_name):
    model.load_weights(f"{model_name}")

# Load models
cnn_model = create_cnn_model(vocab_size, embedding_dim, max_length, filters, kernel_size, dense_dim)
load_model_weights(cnn_model, "Weights_CNN.h5")

mlp_model = create_mlp_model(embedding_dim, dense_dim)
load_model_weights(mlp_model, "Weights_MLP.h5")

lstm_model = create_lstm_model(vocab_size, embedding_dim, max_length, filters, dense_dim)
load_model_weights(lstm_model, "Weights_LSTM.h5")

# Load speech recognition
r = sr.Recognizer()
result = ""

# Perform speech recognition
with sr.Microphone() as source:
    print("Start Speaking")
    recording = r.listen(source)
    print("Times out")

    try:
        result = r.recognize_google(recording, language="id-ID")
        print(result)
    except sr.UnknownValueError:
        print("There's an error, please try again")
    except Exception as e:
        print(e)

# Convert the result to a token sequence
token_list = tokenizer.texts_to_sequences([result])[0]

# Pad the sequence
token_list = pad_sequences([token_list], maxlen=max_length, padding='pre')

probabilities_cnn = cnn_model.predict(token_list)
probabilities_mlp = mlp_model.predict(token_list)
probabilities_lstm = lstm_model.predict(token_list)

# Interpret the prediction results
sentiment_cnn = "Positive" if probabilities_cnn > 0.5 else "Negative"
sentiment_mlp = "Positive" if probabilities_mlp > 0.5 else "Negative"
sentiment_lstm = "Positive" if probabilities_lstm > 0.5 else "Negative"

# Display the prediction results
print("CNN - Sentiment Prediction:", sentiment_cnn, probabilities_cnn)
print("MLP - Sentiment Prediction:", sentiment_mlp, probabilities_mlp)
print("LSTM - Sentiment Prediction:", sentiment_lstm, probabilities_lstm)