import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from collections import namedtuple
from readers.reader import Reader
from collections import Counter

from text_preprocessing import vectorizer,preprocessing
from text_preprocessing.vectorizer import BERT_PREPROCESSING
from model import JOINT_TEXT_MODEL

print("TensorFlow Version:",tf.__version__)
print("Hub version: ",hub.__version__)
print('GPU is in use:',tf.config.list_physical_devices('GPU'))

configuration_file_path = 'config.yaml'
config = {}
config.update(Reader.read_yaml_from_file(configuration_file_path))

data_path = config['data_path']

print('read data ...')
train_text_arr, train_tags_arr, train_intents = Reader.read(data_path+'train/')
val_text_arr, val_tags_arr, val_intents = Reader.read(data_path+'valid/')
data_text_arr, data_tags_arr, data_intents = Reader.read(data_path+'test/')

train_text_arr = preprocessing.remove_next_line(train_text_arr)
train_tags_arr = preprocessing.remove_next_line(train_tags_arr)
train_intents = preprocessing.remove_next_line(train_intents)
print('train_text_arr', len(train_text_arr))

val_text_arr = preprocessing.remove_next_line(val_text_arr)
val_tags_arr = preprocessing.remove_next_line(val_tags_arr)
val_intents = preprocessing.remove_next_line(val_intents)
print('val_text_arr', len(val_text_arr))

data_text_arr = preprocessing.remove_next_line(data_text_arr)
data_tags_arr = preprocessing.remove_next_line(data_tags_arr)
data_intents = preprocessing.remove_next_line(data_intents)
print('Test data size :',len(data_text_arr))

class_dist = Counter(train_intents)
print('Intents & Distributions:',class_dist)

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

# initializing the model
model = JOINT_TEXT_MODEL(slots_num=slots_num,intents_num=intents_num,model_path=config['model_path'],learning_rate=config['LEARNING_RATE'])

# initializing the model tokenizer to be used for creating sub-tokens
model_tokenizer = BERT_PREPROCESSING(model_layer=model.model_layer,max_seq_length=config['MAX_SEQ_LEN'])

print(f'creating input arrays for the model inputs..')
train = model_tokenizer.create_input_array(train_text_arr)
val = model_tokenizer.create_input_array(val_text_arr)

train_tags = np.array([model_tokenizer.get_tag_labels(text,tag_labels,slot_encoder) \
                       for (text,tag_labels) in zip(train_text_arr,train_tags_arr)])
val_tags = np.array([model_tokenizer.get_tag_labels(text,tag_labels,slot_encoder) \
                     for (text,tag_labels) in zip(val_text_arr,val_tags_arr)])

model.fit(train,[train_tags,train_sequence_labels],validation_data=(val,[val_tags,val_sequence_labels]),
          epochs=config['EPOCHS'],batch_size=config['BATCH_SIZE'])

# Model evaluation
query = 'could you please play songs from james blunt'
test_inputs=model_tokenizer.create_input_array([query])
slots,intent=model.predict(test_inputs)
print(f'Test query intent prediction:{sequence_label_encoder.inverse_transform([np.argmax(intent)])}')

# Use the highest logit values for tag prediction
slots=np.argmax(slots, axis=-1)
list_without_pad=[item for sublist in slots for item in sublist if item > 0]
# Removing CLS and SEP tokens from the prediction
pred_tags=slot_encoder.inverse_transform(list_without_pad[1:-1])
print(f'test query - {query}')
print(f'Test query entities prediction:\n{pred_tags}')
print(f'token level entity predictions:{[(word,tag) for word,tag in zip(model_tokenizer.tokenizer.tokenize(query),pred_tags)]}')

print(f"Saving model and its config here - {os.path.join(config['saved_model_dir_path'],config['model_name'],config['model_version'])}")
model.save(os.path.join(config['saved_model_dir_path'],config['model_name'],config['model_version']))
