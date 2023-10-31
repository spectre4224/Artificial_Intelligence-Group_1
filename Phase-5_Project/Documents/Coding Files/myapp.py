from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name)
model = load_model("chatbot_rnn_model.h5")

# Load the tokenizer used for preprocessing
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(conversations)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form['user_message']

    # Preprocess user's input
    input_sequence = tokenizer.texts_to_sequences([user_message])[0]
    input_sequence = pad_sequences([input_sequence], maxlen=max_sequence_length-1, padding='pre')

    # Generate a response using the RNN model
    predicted_word_index = np.argmax(model.predict(input_sequence))
    response = tokenizer.index_word[predicted_word_index]

    return response

if __name__ == '__main__':
    app.run(debug=True)
