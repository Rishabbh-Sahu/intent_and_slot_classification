# -*- coding: utf-8 -*-
"""
Created on Mon Aug 09 2021
@author: Rishabbh sahu
"""

import os
import pickle
import numpy as np
import json

from readers.reader import Reader
from text_preprocessing.vectorizer import BERT_PREPROCESSING_FAST
from model import JOINT_TEXT_MODEL
from flask import Flask, jsonify, request

# Create app
app = Flask(__name__)

def initialize():    
    global model_tokenizer
    # initializing the model tokenizer to be used for creating sub-tokens
    model_tokenizer = BERT_PREPROCESSING_FAST(max_seq_length=config['MAX_SEQ_LEN'])

    # loading models
    print('Loading models and artifacts...')
    if not os.path.exists(load_folder_path):
        print('Folder `%s` not exist' % load_folder_path)

    global slot_encoder
    with open(os.path.join(load_folder_path, 'slot_label_encoder.pkl'), 'rb') as handle:
        slot_encoder = pickle.load(handle)

    global sequence_label_encoder
    with open(os.path.join(load_folder_path, 'sequence_label_encoder.pkl'), 'rb') as handle:
        sequence_label_encoder = pickle.load(handle)

    global model
    with open(os.path.join(load_folder_path, 'model_params.json'), 'r') as json_file:
        model_params = json.load(json_file)
    model = JOINT_TEXT_MODEL(slots_num=model_params['num_slot_classes'],intents_num=model_params['num_sequence_classes'],
                             model_path=model_params['model_path'],learning_rate=model_params['learning_rate'])
    model.load(load_folder_path)

@app.route('/', methods=['GET', 'POST'])
def hello():
    return 'Hello from NLU inference routine'

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    input_json = request.json
    utterance = input_json["utterance"]

    encodings = model_tokenizer.fastTokenizer([utterance.split()], is_split_into_words=True)
    input_txt = model_tokenizer.create_model_input(encodings)
    slots, intent = model.predict(input_txt)
    predicted_intent = sequence_label_encoder.inverse_transform([np.argmax(intent)])
    slots = np.argmax(slots, axis=-1)
    list_without_pad = [item for sublist in slots for item in sublist if item > 0]
    # Removing CLS and SEP tokens from the prediction
    pred_tags = slot_encoder.inverse_transform(list_without_pad[1:-1])
    annotations = [{word:tag} for word, tag in zip(model_tokenizer.fastTokenizer.tokenize(utterance), pred_tags)]

    response = {
        "intent": {
            "name": str(predicted_intent[0]),
        },
        "algo": "Joint text model",
        "annotations": annotations
    }
    return jsonify(response)

if __name__ == '__main__':

    configuration_file_path = 'config.yaml'
    config = {}
    config.update(Reader.read_yaml_from_file(configuration_file_path))
    load_folder_path = os.path.join(config['saved_model_dir_path'], config['model_name'], config['model_version'])

    print(('Starting the Server'))
    initialize()
    # Run app
    app.run(host='0.0.0.0', port=8888, debug=False, use_reloader=False)
