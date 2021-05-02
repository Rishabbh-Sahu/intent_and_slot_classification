# -*- coding: utf-8 -*-
"""
Created on tuesday 13-Apr-2021
@author: rishabbh-sahu
"""

import os
import tensorflow as tf
import numpy as np
from readers.reader import Reader
from collections import Counter

from text_preprocessing import vectorizer
from text_preprocessing.vectorizer import BERT_PREPROCESSING_FAST
from model import JOINT_TEXT_MODEL

print("system specification...")
print("TensorFlow Version:",tf.__version__)
print('GPU is in use:',tf.config.list_physical_devices('GPU'))

configuration_file_path = 'config.yaml'
config = {}
config.update(Reader.read_yaml_from_file(configuration_file_path))

data_path = config['data_path']

print('read data ...')
train_text_arr, train_tags_arr, train_intents = Reader.read(data_path+'train/')
val_text_arr, val_tags_arr, val_intents = Reader.read(data_path+'valid/')
data_text_arr, data_tags_arr, data_intents = Reader.read(data_path+'test/')

print('train_text_arr', len(train_text_arr))
print('val_text_arr', len(val_text_arr))
print('Test data size :',len(data_text_arr))

class_dist = Counter(train_intents)
print('classes & distributions:',class_dist)

print('encode sequence labels ...')
sequence_label_encoder = vectorizer.label_encoder(train_intents)
train_sequence_labels = vectorizer.label_encoder_transform(train_intents,sequence_label_encoder)
val_sequence_labels = vectorizer.label_encoder_transform(val_intents,sequence_label_encoder)
intents_num = len(sequence_label_encoder.classes_);print(f'total number of sequence labels are {intents_num}.')

print('encode sequence tags ...')
tags_data = ['<PAD>'] + [item for sublist in [s.split() for s in train_tags_arr] for item in sublist] \
                       + [item for sublist in [s.split() for s in val_tags_arr] for item in sublist]
slot_encoder = vectorizer.label_encoder(tags_data)
slots_num = len(slot_encoder.classes_);print(f'total number of slots are {slots_num}.')

print('data pre-processing...')
train_text_arr = [t.split() for t in train_text_arr]
train_tags_arr = [t.split() for t in train_tags_arr]

val_text_arr = [t.split() for t in val_text_arr]
val_tags_arr = [t.split() for t in val_tags_arr]

# initializing the model
model = JOINT_TEXT_MODEL(slots_num=slots_num,intents_num=intents_num,model_path=config['model_path'],learning_rate=config['LEARNING_RATE'])

# initializing the model tokenizer to be used for creating sub-tokens
model_tokenizer = BERT_PREPROCESSING_FAST(max_seq_length=config['MAX_SEQ_LEN'])

print(f'creating input arrays for the model inputs..')
train_encodings = model_tokenizer.fastTokenizer(train_text_arr, is_split_into_words=True,max_length=config['MAX_SEQ_LEN'],
                                                padding=True,return_offsets_mapping=True,truncation=True)
val_encodings = model_tokenizer.fastTokenizer(val_text_arr, is_split_into_words=True,max_length=config['MAX_SEQ_LEN'],
                                              padding=True,return_offsets_mapping=True,truncation=True)

train = model_tokenizer.create_model_input(train_encodings)
val = model_tokenizer.create_model_input(val_encodings)

train_labels = model_tokenizer.encode_tags(train_tags_arr, train_encodings,slot_encoder)
train_tags = np.array(train_labels)

val_labels = model_tokenizer.encode_tags(val_tags_arr, val_encodings,slot_encoder)
val_tags = np.array(val_labels)

print('training started ...')
tf.keras.backend.clear_session()
model.fit(train,[train_tags,train_sequence_labels],validation_data=(val,[val_tags,val_sequence_labels]),
          epochs=config['EPOCHS'],batch_size=config['BATCH_SIZE'])
print('training completed ...')

# Model evaluation
query = 'could you please play songs from james blunt'
test_encodings = model_tokenizer.fastTokenizer([query.split()],is_split_into_words=True)
test_inputs=model_tokenizer.create_model_input(test_encodings)
slots,intent=model.predict(test_inputs)
print(f'Test query intent prediction:{sequence_label_encoder.inverse_transform([np.argmax(intent)])}')

# Use the highest logit values for tag prediction
slots = np.argmax(slots, axis=-1)
list_without_pad = [item for sublist in slots for item in sublist if item > 0]
# Removing CLS and SEP tokens from the prediction
pred_tags = slot_encoder.inverse_transform(list_without_pad[1:-1])
print('Test query entities prediction:\n',pred_tags)
print(f'test query - {query}')
print(f'Test query entities prediction:\n{pred_tags}')
print(f'token level entity predictions:{[(word,tag) for word,tag in zip(model_tokenizer.fastTokenizer.tokenize(query),pred_tags)]}')

print(f"Saving model and its config here - {os.path.join(config['saved_model_dir_path'],config['model_name'],config['model_version'])}")
model.save(os.path.join(config['saved_model_dir_path'],config['model_name'],config['model_version']))
