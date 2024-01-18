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
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
list_stopwords = set(stopwords.words('indonesian'))

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

# Definisikan fungsi untuk cleansing
df_alay = pd.read_csv('new_kamusalay.csv', encoding='ISO-8859-1', header=None)
df_alay = df_alay.rename(columns={0: 'alay', 1: 'formal'}) 

def data_cleaning (text):
    clean1 = re.sub ('\\n','', text)
    clean2 = re.sub ('RT',' ', clean1)
    clean3 = re.sub ('USER', ' ', clean2)
    clean4 = re.sub ('(http|https):\/\/s+', ' ', clean3)
    clean5 = re.sub ('[^0-9a-zA-Z]+', ' ', clean4)
    clean6 = re.sub ('x[a-z0-9]{2}', ' ', clean5)
    clean7 = re.sub ("\d+", ' ', clean6)
    clean8 = re.sub ('  +', '', clean7)
    return clean8

def case_folding (text):
    return text.lower()

def alay_normalization(text):
    res = ''
    for item in text.split():
        if item in df_alay['alay'].values:
            res += df_alay[df_alay['alay'] == item]['formal'].iloc[0]
        else:
            res += item
        res += ' '
    return res

def stopword_removal(text):
    resp = ''
    for item in text.split():
        if item not in list_stopwords:
            resp += item
        resp +=' '
    clean = re.sub('  +', ' ', resp)
    return clean

def cleansing(text):
    text = data_cleaning(text)
    text = case_folding(text)
    text = alay_normalization(text)
    text = stopword_removal(text)
    return text

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

model_file_from_lstm = load_model('data/model_lstm.keras')

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
    text = [cleansing(original_text)]

    # convert text to vector
    feature = feature_file_from_nn.transform(text)

    # predict sentiment
    get_sentiment = model_file_from_nn.predict(feature)[0]

    # return response    
    json_response = {
        'status_code': 200,
        'description': "Result of Sentiment Analysis Using NN",
        'data': {
            'text': original_text,
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
    df['text_clean'] = df.apply(lambda row : cleansing(row['text']), axis = 1)
    
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
        'status_code' : 200,
        'description' : "Result of Sentiment Analysis using LSTM",
        'data' : {
            'text' : original[:3],
            'sentiment' : result[:3],
            'keterangan' : 'File lengkap telah disimpan dalam folder output.'
        },
    }

    response_data = jsonify(json_response)
    return response_data
##################################################################################
# Endpoint LSTM teks

@swag_from('docs/LSTM_text.yml',methods=['POST'])
@app.route('/LSTM_text',methods=['POST'])
def lstm_text():  
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

# Endpoint LSTM file
@swag_from('docs/LSTM_file.yml',methods=['POST'])
@app.route('/LSTM_file',methods=['POST'])
def lstm_file():
    file = request.files["upload_file"]
    df = (pd.read_csv(file, encoding="latin-1"))
    df = df.rename(columns={df.columns[0]: 'text'})
    df['text_clean'] = df.apply(lambda row : cleansing(row['text']), axis = 1)
    
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
        'status_code' : 200,
        'description' : "Result of Sentiment Analysis using LSTM",
        'data' : {
            'text' : original[:3],
            'sentiment' : result[:3],
            'keterangan' : 'File lengkap telah disimpan dalam folder output.'
        },
    }
    response_data = jsonify(json_response)
    return response_data

##################################################################################
if __name__ == '__main__':
    app.run()