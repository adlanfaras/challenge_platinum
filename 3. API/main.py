from flask import Flask, jsonify, request #import objects from the Flask model
from flasgger import Swagger, LazyString, LazyJSONEncoder
from flasgger import swag_from
import numpy as np
import pickle

import pandas as pd
import sqlite3

import tensorflow
tensorflow.config.experimental.list_physical_devices('GPU')
from keras.models import load_model
from keras.preprocessing.text import tokenizer_from_json
from keras.utils.data_utils import pad_sequences

#Memanggil fungsi yang ada di file proses.py
from proses import process_csv_nn, process_csv_lstm, cleansing, pred_sentiment, pred

app = Flask(__name__) #define app using Flask
app.json_encoder = LazyJSONEncoder

swagger_template = dict(
    info = {
        'title': LazyString(lambda:'API Data Science Platinum Challenge'),
        'version': LazyString(lambda:'1.0.0'),
        'description': LazyString(lambda:'Dokumentasi Data Science Class Platinum Challenge')
        }, host = LazyString(lambda: request.host)
)

swagger_config = {
        "headers":[],
        "specs":[
            {
            "endpoint":'docs',
            "route":'/docs.json'
            }
        ],
        "static_url_path":"/flasgger_static",
        "swagger_ui":True,
        "specs_route":"/docs/"
}
swagger = Swagger(app, template=swagger_template, config=swagger_config)

db = sqlite3.connect('dbp.db', check_same_thread=False)
db.text_factory = bytes
mycursor = db.cursor()

#POST METHOD MODEL NEURAL NETWORK
@swag_from("docs/post_text_nn.yml", methods=['POST'])
@app.route('/post_text_nn', methods=['POST'])
def postTextNN():
    string = str(request.form["text"])
    
    with open('Model/model_nn.pkl', 'rb') as f: 
        model_nn = pickle.load(f)

    with open('Model/countvect.pkl', 'rb') as g: 
        count_vect_nn = pickle.load(g)
    string = cleansing(string)
    text = count_vect_nn.transform([string])

    result = model_nn.predict(text)[0]
    hasil=str(result)

    query_tabel = "insert into prediksi_tweet (tweet,prediksi) values (?, ?)"
    value = (string, hasil)
    mycursor.execute(query_tabel, value)
    db.commit()

    return  f"Hasil adalah {hasil}"


@swag_from("docs/post_file_nn.yml", methods=['POST'])
@app.route('/post_file_nn', methods=['POST'])
def postFileNN():
    file = request.files['file']
    try:
        data = pd.read_csv(file, encoding="latin-1")
    except:
        data = pd.read_csv(file, encoding='utf-8',error_bad_lines=False)

        first_column = input_file.iloc[:, 0]
    
    print(first_column)
    with open('model_of_nn/model_nn.pkl', 'rb') as f: 
        model_nn = pickle.load(f)

    with open('resource_of_lstm/vectorizer.pkl', 'rb') as g: 
        count_vect_nn = pickle.load(g)

    for tweet in first_column:
        tweet = cleansing(tweet)
        text = count_vect_nn.transform([tweet])
        result = model_nn.predict(text)[0]
        hasil=str(result)

    json_response = {
            'status_code': 200,
            'description': "Result of Sentiment Analysis using NN",
            'data': {
                'text': tweet,
                'sentiment': hasil
            },
        }
        response_data = jsonify(json_response)
        return response_data
#=======================================================================================#
#POST METHOD MODEL LSTM
@swag_from("docs/post_text_lstm.yml", methods=['POST'])
@app.route('/post_text_lstm', methods=['POST'])
def type():
    string = str(request.form["text"])
    string = cleansing(string)
    pred_sentiment(string)

    classes = pred_sentiment(string)
    hasil = pred(classes)

    query_tabel = "insert into prediksi_tweet (tweet,prediksi) values (?, ?)"
    value = (string, hasil)
    mycursor.execute(query_tabel, value)
    db.commit()

    return  f"Hasil adalah {hasil}"

@swag_from("docs/post_file_lstm.yml", methods=['POST'])
@app.route('/post_file_lstm', methods=['POST'])
def postFileLSTM():
    file = request.files['file']
    try:
        data = pd.read_csv(file, encoding='iso-8859-1', error_bad_lines=False)
    except:
        data = pd.read_csv(file, encoding='utf-8', error_bad_lines=False) 
    process_csv_lstm(data)
    return "DONE"


if __name__ == '__main__':
    app.run() #run app on port 8080 in debug mode
