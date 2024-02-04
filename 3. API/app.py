from flask import Flask, jsonify
from flask import request
from flasgger import Swagger, LazyString, LazyJSONEncoder
from flasgger import swag_from

import pickle, re
import sqlite3
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from cleansing import cleansing_all

class CustomFlaskAppWithEncoder(Flask):
    json_provider_class = LazyJSONEncoder

app = CustomFlaskAppWithEncoder(__name__)
app.json_encoder = LazyJSONEncoder
swagger_template = dict(
info = {
    'title': LazyString(lambda: 'API Documentation for Deep Learning'),
    'version': LazyString(lambda: '1.0.0'),
    'description': LazyString(lambda: 'Challenge Level Platinum Binar Academy'),
    },
    host = LazyString(lambda: request.host)
)
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'docs',
            "route": '/docs.json',
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs/"
}
swagger = Swagger(app, template=swagger_template, config=swagger_config)

##################################################################################
# Definisikan parameter untuk feature extraction
max_features = 100000
tokenizer = Tokenizer(num_words=max_features, split=' ',lower=True)

# Definisikan label untuk sentimen
sentiment = ['negative', 'neutral', 'positive']

##################################################################################
# Load feature extraction and model Neural Network
file = open("data/vectorizer.pkl",'rb')
feature_file_from_nn = pickle.load(file)

model_file_from_nn = pickle.load(open('data/model_nn.pkl', 'rb'))

# Load feature extraction and model LSTM
file = open("data/x_pad_sequences.pickle",'rb')
feature_file_from_lstm = pickle.load(file)
file = open("data/tokenizer.pickle",'rb')
tokenizer_from_lstm = pickle.load(file)
file.close()

model_file_from_lstm = load_model('data/model_lstm.h5')

##################################################################################

#endpoint
@swag_from("docs/hello_world.yml", methods=['GET'])
@app.route('/', methods=['GET'])
def hello_world():
    json_response = {
        'status_code': 200,
        'description': "API untuk Deep Learning",
        'data': "Kelompok 4 : Binar Academy, Data Science Gelombang 14",}

    response_data = jsonify(json_response)
    return response_data


##################################################################################
# Endpoint NN teks
@swag_from('docs/nn_text.yml',methods=['POST'])
@app.route('/nn_text',methods=['POST'])
def nn_text():  
    # get input text
    original_text = request.form.get('text')
    text = [cleansing_all(original_text)]

    # convert text to vector
    feature = feature_file_from_nn.transform(text)

    # predict sentiment
    get_sentiment = model_file_from_nn.predict(feature)[0]

    # return response    
    json_response = {
        'status_code': 200,
        'description': "Result of Sentiment Analysis Using NN",
        'data': {
            'text': text,
            'sentiment': get_sentiment}
                    }

    response_data = jsonify(json_response)
    return response_data

# Endpoint NN file
@swag_from('docs/nn_file.yml',methods=['POST'])
@app.route('/nn_file',methods=['POST'])
def nn_file():
    file = request.files["upload_file"]
    df = (pd.read_csv(file, encoding="latin-1"))
    df = df.rename(columns={df.columns[0]: 'text'})
    df['text_clean'] = df.apply(lambda row : cleansing_all(row['text']), axis = 1)
    
    result = []

    for index, row in df.iterrows():
        text = feature_file_from_nn.transform([row['text_clean']])
        prediction = model_file_from_nn.predict(text)[0]
        result.append(prediction)
        original = df.text_clean.to_list()

    db_path1 = 'output/prediksi_nn.db'
    conn = sqlite3.connect(db_path1)
    cursor = conn.cursor()

    cursor.execute('''
            CREATE TABLE IF NOT EXISTS tabel_prediksi (
                text TEXT,
                text_clean TEXT,
                result INTEGER
            )
        ''')
    
    for index, row in df.iterrows():
        cursor.execute('''
                INSERT INTO tabel_prediksi (text, text_clean, result)
                VALUES (?, ?, ?)
            ''', (row['text'], row['text_clean'], result[index]))

    conn.commit()
    conn.close()

    json_response = {
    'status_code': 200,
    'description': 'File lengkap telah disimpan dalam folder output.',
    'data': [
        {'text': original[0],
        'sentiment': result[0],
        'keterangan': "Hasil dari Sentiment Analysis menggunakan NN" },
        {'text': original[1],
         'sentiment': result[1],},
        {'text': original[2],
         'sentiment': result[2],}]}

    response_data = jsonify(json_response)
    return response_data
##################################################################################
# Endpoint LSTM teks

@swag_from('docs/LSTM_text.yml',methods=['POST'])
@app.route('/LSTM_text',methods=['POST'])
def lstm_text():  
    # get input text and cleansing
    original_text = request.form.get('text')
    text = [cleansing_all(original_text)]

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
            'text': text,
            'sentiment': get_sentiment
        }
    }

    response_data = jsonify(json_response)
    return response_data

# Endpoint LSTM file
@swag_from('docs/LSTM_file.yml',methods=['POST'])
@app.route('/LSTM_file',methods=['POST'])
def lstm_file():
    file = request.files["upload_file"]
    df = (pd.read_csv(file, encoding="latin-1"))
    df = df.rename(columns={df.columns[0]: 'text'})
    df['text_clean'] = df.apply(lambda row : cleansing_all(row['text']), axis = 1)
    
    result = []

    for index, row in df.iterrows():
        text = tokenizer_from_lstm.texts_to_sequences([(row['text_clean'])])
        guess = pad_sequences(text, maxlen=feature_file_from_lstm.shape[1])
        prediction = model_file_from_lstm.predict(guess)
        polarity = np.argmax(prediction[0])
        get_sentiment = sentiment[polarity]
        result.append(get_sentiment)

    original = df.text_clean.to_list()

    db_path2 = 'output/prediksi_lstm.db'
    conn = sqlite3.connect(db_path2)
    cursor = conn.cursor()

    cursor.execute('''
            CREATE TABLE IF NOT EXISTS tabel_prediksi (
                text TEXT,
                text_clean TEXT,
                result INTEGER
            )
        ''')
    
    for index, row in df.iterrows():
        cursor.execute('''
                INSERT INTO tabel_prediksi (text, text_clean, result)
                VALUES (?, ?, ?)
            ''', (row['text'], row['text_clean'], result[index]))

    conn.commit()
    conn.close()

    json_response = {
    'status_code': 200,
    'description': 'File lengkap telah disimpan dalam folder output.',
    'data': [
        {'text': original[0],
        'sentiment': result[0],
        'keterangan': "Hasil dari Sentiment Analysis menggunakan LSTM" },
        {'text': original[1],
         'sentiment': result[1],},
        {'text': original[2],
         'sentiment': result[2],}]}

    response_data = jsonify(json_response)
    return response_data

##################################################################################
if __name__ == '__main__':
    app.run()