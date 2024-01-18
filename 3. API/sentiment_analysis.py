import re, pickle
import numpy as np

from flask import Flask, jsonify
from flask import request
import flask
from flasgger import Swagger, LazyString, LazyJSONEncoder
from flasgger import swag_from

from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

class CustomFlaskAppWithEncoder(Flask):
    json_provider_class = LazyJSONEncoder

app = CustomFlaskAppWithEncoder(__name__)

swagger_template = dict(
    info = {
        'title' : LazyString(lambda: "API Documentation for Deep Learning"),
        'version' : LazyString(lambda: "1.0.0"),
        'description' : LazyString(lambda: "Dokumentasi API untuk  Deep Learning"),
    },
    host = LazyString(lambda: request.host)
)

swagger_config = {
    "headers" : [],
    "specs" : [
        {
            "endpoint": "docs",
            "route" : "/docs.json",
        }
    ],
    "static_url_path": "/flasgger_static",
    # "static_folder": "static",  # must be set by user
    "swagger_ui": True,
    "specs_route": "/docs/"
}
swagger = Swagger(app, template=swagger_template, config = swagger_config)

# Initialize variable
max_features = 100000
sentiment = ['negative','neutral','positive']

# Function text cleaning
def cleansing(sent):
    string = sent.lower()
    string = re.sub(r'[^a-zA-Z0-9]', ' ', string)
    return string

# Load feature extraction and model Neural Network
file = open("resources_of_nn/tfidf_vect.p",'rb')
feature_file_from_nn = pickle.load(file)

model_file_from_nn = pickle.load(open('model_of_nn/model_neuralnetwork.p', 'rb'))

# Load feature extraction and model CNN
file = open("resources_of_cnn/x_pad_sequences.pickle",'rb')
feature_file_from_cnn = pickle.load(file)
file = open("resources_of_cnn/tokenizer.pickle",'rb')
tokenizer_from_cnn = pickle.load(file)
file.close()

model_file_from_cnn = load_model('model_of_cnn/model_cnn.keras')

# Load feature extraction and model RNN
file = open("resources_of_rnn/x_pad_sequences.pickle",'rb')
feature_file_from_rnn = pickle.load(file)
file = open("resources_of_rnn/tokenizer.pickle",'rb')
tokenizer_from_rnn = pickle.load(file)
file.close()

model_file_from_rnn = load_model('model_of_rnn/model_rnn.keras')

# Load feature extraction and model LSTM
file = open("resources_of_lstm/x_pad_sequences.pickle",'rb')
feature_file_from_lstm = pickle.load(file)
file = open("resources_of_lstm/tokenizer.pickle",'rb')
tokenizer_from_lstm = pickle.load(file)
file.close()

model_file_from_lstm = load_model('model_of_lstm/model_lstm.keras')

# API Sentiment Analysis using Neural Network
@swag_from("docs/nn.yml", methods = ['POST'])
@app.route('/nn', methods=['POST'])
def nn():

    # get input text
    original_text = request.form.get('text')

    # convert text to vector
    text = feature_file_from_nn.transform([cleansing(original_text)])

    # predict sentiment
    get_sentiment = model_file_from_nn.predict(text)[0]

    # return response    
    json_response = {
        'status_code': 200,
        'description': "Result of Sentiment Analysis Using CNN",
        'data': {
            'text': original_text,
            'sentiment': get_sentiment
        }
    }

    response_data = jsonify(json_response)
    return response_data

# API Sentiment Analysis using CNN
@swag_from("docs/cnn.yml", methods = ['POST'])
@app.route('/cnn', methods=['POST'])
def cnn():

    # get input text and cleansing
    original_text = request.form.get('text')
    text = [cleansing(original_text)]

    # convert text to vector
    feature = tokenizer_from_cnn.texts_to_sequences(text)
    feature = pad_sequences(feature, maxlen=feature_file_from_cnn.shape[1])

    # predict sentiment
    prediction = model_file_from_cnn.predict(feature)
    get_sentiment = sentiment[np.argmax(prediction[0])]

    # return response
    json_response = {
        'status_code': 200,
        'description': "Result of Sentiment Analysis Using CNN",
        'data': {
            'text': original_text,
            'sentiment': get_sentiment
        }
    }

    response_data = jsonify(json_response)
    return response_data

# API Sentiment Analysis using RNN
@swag_from("docs/rnn.yml", methods = ['POST'])
@app.route('/rnn', methods=['POST'])
def rnn():

    # get input text and cleansing
    original_text = request.form.get('text')
    text = [cleansing(original_text)]

    # convert text to vector
    feature = tokenizer_from_rnn.texts_to_sequences(text)
    feature = pad_sequences(feature, maxlen=feature_file_from_rnn.shape[1])

    # predict sentiment
    prediction = model_file_from_rnn.predict(feature)
    get_sentiment = sentiment[np.argmax(prediction[0])]

    # return response 
    json_response = {
        'status_code': 200,
        'description': "Result of Sentiment Analysis Using RNN",
        'data': {
            'text': original_text,
            'sentiment': get_sentiment
        }
    }

    response_data = jsonify(json_response)
    return response_data

# API Sentiment Analysis using LSTM
@swag_from("docs/lstm.yml", methods = ['POST'])
@app.route('/lstm', methods=['POST'])
def lstm():

    # get input text and cleansing
    original_text = request.form.get('text')
    text = [cleansing(original_text)]

    # convert text to vector
    feature = tokenizer_from_lstm.texts_to_sequences(text)
    feature = pad_sequences(feature, maxlen=feature_file_from_lstm.shape[1])

    # predict sentiment
    prediction = model_file_from_lstm.predict(feature)
    get_sentiment = sentiment[np.argmax(prediction[0])]

    # return response    
    json_response = {
        'status_code': 200,
        'description': "Result of Sentiment Analysis Using LSTM",
        'data': {
            'text': original_text,
            'sentiment': get_sentiment
        }
    }

    response_data = jsonify(json_response)
    return response_data

if __name__ == '__main__':
    app.run()

