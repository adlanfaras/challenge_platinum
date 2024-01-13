import re
import json
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

db = sqlite3.connect('dbp.db', check_same_thread=False)
db.text_factory = bytes
mycursor = db.cursor()

#Baca Table Kamus Alay
query = "select * from kamus_alay"
kamusalay = pd.read_sql_query(query, db)
kamusalay['hasil clean'] = kamusalay['hasil clean'].str.decode('utf-8')
kamusalay['kata alay'] = kamusalay['kata alay'].str.decode('utf-8')

#Mereplace kamus alay
alay_dict_map = dict(zip(kamusalay['kata alay'], kamusalay['hasil clean']))
def replace_kamus_alay(text):
    for word in alay_dict_map:
        return ' '.join([alay_dict_map[word] if word in alay_dict_map else word for word in text.split(' ')])

#Ubah text menjadi lower
def lower(text):
    return text.lower()

#Hapus karakter pada text
def hapuskarakter(text):
    text = re.sub('\n',' ', text)
    text = re.sub('rt',' ', text)
    text = re.sub('user',' ', text)
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))',' ',text)
    text = re.sub('  +',' ', text)
    return text

#Menjalankan semua proses cleansing
def cleansing(text):
    text = lower(text)
    text = hapuskarakter(text)
    text = replace_kamus_alay(text)
    return text

#PREDIKSI SENTIMEN UNTUK LSTM
def pred_sentiment(string):
    with open('Model/tokenizer.json') as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)

    loaded_model = tensorflow.keras.models.load_model('Model/Model_LSTM.h5', compile=False)

    string = cleansing(string)
    text = [string]

    sekuens_x = tokenizer.texts_to_sequences(text)
    padded_x = pad_sequences(sekuens_x)

    classes = loaded_model.predict(padded_x, batch_size=10)

    return classes[0]

#MENENTUKAN KELAS HASIL PREDIKSI
def pred(classes):
    hasil=""
    if classes[0] == classes.max():
        hasil = "negative"
        return hasil
    if classes[1] == classes.max():
        hasil = "neutral"
        return hasil
    if classes[2] == classes.max():
        hasil = "positif"
        return hasil

#PROSES UNTUK INPUT FILE NEURAL NETWORK
def process_csv_nn(input_file):
    first_column = input_file.iloc[:, 0]
    print(first_column)
    with open('Model/model_nn.pkl', 'rb') as f: 
        model_nn = pickle.load(f)

    with open('Model/countvect.pkl', 'rb') as g: 
        count_vect_nn = pickle.load(g)

    for tweet in first_column:
        tweet = cleansing(tweet)
        text = count_vect_nn.transform([tweet])
        result = model_nn.predict(text)[0]
        hasil=str(result)

        query_tabel = "insert into prediksi_tweet (tweet,prediksi) values (?, ?)"
        value = (tweet, hasil)
        mycursor.execute(query_tabel, value)
        db.commit()
        print(tweet)

#PROSES UNTUK INPUT FILE LSTM
def process_csv_lstm(input_file):
    first_column = input_file.iloc[:, 0]
    print(first_column)

    for tweet in first_column:
        tweet = cleansing(tweet)
        pred_sentiment(tweet)
        classes = pred_sentiment(tweet)
        hasil = pred(classes)

        query_tabel = "insert into prediksi_tweet (tweet,prediksi) values (?, ?)"
        value = (tweet, hasil)
        mycursor.execute(query_tabel, value)
        db.commit()
        print(tweet)
