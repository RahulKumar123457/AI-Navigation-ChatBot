from flask import Flask, render_template, request, jsonify
import numpy as np
from keras.models import load_model
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle

app = Flask(__name__)

# Load pre-trained model and other necessary data
model = load_model('model.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
lemmatizer = WordNetLemmatizer()

# Function to preprocess user input
def preprocess_input(text):
    return [lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(text)]

# Function to predict intent based on user input
def predict_intent(text):
    input_words = preprocess_input(text)
    input_bag = [1 if word in input_words else 0 for word in words]
    input_bag = np.array(input_bag).reshape(1, -1)
    result = model.predict(input_bag)[0]
    predicted_class_index = np.argmax(result)
    return classes[predicted_class_index]

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')


# Changes....
@app.route('/webhook', methods=['POST'])
def webhook():
    req = request.get_json(force=True)
    print(req)

    return {
        'fulfillmentText': 'Hello from the other side.'
    }
    
# Define route for processing user input
@app.route('/process', methods=['POST'])
def process():
    user_input = request.form['user_input']
    intent = predict_intent(user_input)
    # Here you can handle different intents and provide appropriate responses
    # For simplicity, I'm just returning the predicted intent in JSON format
    return jsonify({'intent': intent})

if __name__ == '__main__':
    app.run(debug=True)