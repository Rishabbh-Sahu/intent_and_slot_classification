import os
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text # A dependency of the preprocessing model
import numpy as np
from readers.goo_format_reader import Reader
from sklearn.preprocessing import LabelEncoder


tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/1'
tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
print(f'BERT model selected           : {tfhub_handle_encoder}')
print(f'Preprocessing model auto-selected: {tfhub_handle_preprocess}')

# bert_preprocess = hub.load(tfhub_handle_preprocess)
# tok = bert_preprocess.tokenize(tf.constant(['Hello TensorFlow!']))
# print(tok)
'''
def make_bert_preprocess_model(sentence_features, seq_length=128):
  """Returns Model mapping string features to BERT inputs.

  Args:
    sentence_features: a list with the names of string-valued features.
    seq_length: an integer that defines the sequence length of BERT inputs.

  Returns:
    A Keras Model that can be called on a list or dict of string Tensors
    (with the order or names, resp., given by sentence_features) and
    returns a dict of tensors for input to BERT.
  """

  input_segments = [
      tf.keras.layers.Input(shape=(), dtype=tf.string, name=ft)
      for ft in sentence_features]

  # Tokenize the text to word pieces.
  bert_preprocess = hub.load(tfhub_handle_preprocess)
  tokenizer = hub.KerasLayer(bert_preprocess.tokenize, name='tokenizer')
  segments = [tokenizer(s) for s in input_segments]

  # Optional: Trim segments in a smart way to fit seq_length.
  # Simple cases (like this example) can skip this step and let
  # the next step apply a default truncation to approximately equal lengths.
  truncated_segments = segments

  # Pack inputs. The details (start/end token ids, dict of output tensors)
  # are model-dependent, so this gets loaded from the SavedModel.
  packer = hub.KerasLayer(bert_preprocess.bert_pack_inputs,
                          arguments=dict(seq_length=seq_length),
                          name='packer')
  model_inputs = packer(truncated_segments)
  return tf.keras.Model(input_segments, model_inputs)
'''

def build_classifier_model(num_classes):
  inputs = dict(
      input_word_ids=tf.keras.layers.Input(shape=(None,), dtype=tf.int32),
      input_mask=tf.keras.layers.Input(shape=(None,), dtype=tf.int32),
      input_type_ids=tf.keras.layers.Input(shape=(None,), dtype=tf.int32),
  )

  encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='encoder')
  net = encoder(inputs)['pooled_output']
  net = tf.keras.layers.Dropout(rate=0.1)(net)
  net = tf.keras.layers.Dense(num_classes, activation='softmax', name='intent_classifier')(net)
  return tf.keras.Model(inputs, net, name='prediction')

# text_preprocessed = make_bert_preprocess_model(test_text)
#
# print('Keys           : ', list(text_preprocessed.keys()))
# print('Shape Word Ids : ', text_preprocessed['input_word_ids'].shape)
# print('Word Ids       : ', text_preprocessed['input_word_ids'][0, :16])
# print('Shape Mask     : ', text_preprocessed['input_mask'].shape)
# print('Input Mask     : ', text_preprocessed['input_mask'][0, :16])
# print('Shape Type Ids : ', text_preprocessed['input_type_ids'].shape)
# print('Type Ids       : ', text_preprocessed['input_type_ids'][0, :16])

# test_classifier_model = build_classifier_model(2)
# bert_raw_result = test_classifier_model(text_preprocessed)
# print(tf.sigmoid(bert_raw_result))

# tf.keras.utils.plot_model(test_classifier_model)

print('read data ...')
train_text_arr, train_tags_arr, train_intents = Reader.read('./data/Productivity_Agent_11/train/')
val_text_arr, val_tags_arr, val_intents = Reader.read('./data/Productivity_Agent_11/valid/')
data_text_arr, data_tags_arr, data_intents = Reader.read('./data/Productivity_Agent_11/test/')

### data cleansing in order to remove next line symbol may arise while data preperation
train_text_arr = [sub.replace('\n', '') for sub in train_text_arr]
train_tags_arr = [sub.replace('\n', '') for sub in train_tags_arr]
train_intents = [sub.replace('\n', '') for sub in train_intents]
print('train_text_arr', len(train_text_arr))

val_text_arr = [sub.replace('\n', '') for sub in val_text_arr]
val_tags_arr = [sub.replace('\n', '') for sub in val_tags_arr]
val_intents = [sub.replace('\n', '') for sub in val_intents]
print('val_text_arr', len(val_text_arr))

data_text_arr = [sub.replace('\n', '') for sub in data_text_arr]
data_tags_arr = [sub.replace('\n', '') for sub in data_tags_arr]
data_intents = [sub.replace('\n', '') for sub in data_intents]
print('Test data size :',len(data_text_arr))

print('encode labels ...')
intents_label_encoder = LabelEncoder()
train_intents = intents_label_encoder.fit_transform(train_intents).astype(np.int32)
val_intents = intents_label_encoder.transform(val_intents).astype(np.int32)
intents_num = len(intents_label_encoder.classes_);print('total number of intents:',intents_num)

epochs = 1
batch_size = 16
init_lr = 2e-5
max_seq_length = 128
# text_preprocessed = bert_preprocess.bert_pack_inputs([bert_preprocess.tokenize(tf.constant(train_text_arr))], max_seq_length)
# print('Shape Word Ids : ', text_preprocessed['input_word_ids'].shape)
# print('Word Ids       : ', text_preprocessed['input_word_ids'])
# print('Shape Mask     : ', text_preprocessed['input_mask'].shape)
# print('Input Mask     : ', text_preprocessed['input_mask'])
# print('Shape Type Ids : ', text_preprocessed['input_type_ids'].shape)
# print('Type Ids       : ', text_preprocessed['input_type_ids'])

bert_preprocess = hub.load(tfhub_handle_preprocess)
# print(bert_preprocess.tokenize(tf.constant(train_text_arr[:1])))
print('query:',train_text_arr[:2])
train_dataset = bert_preprocess.bert_pack_inputs([bert_preprocess.tokenize(tf.constant(train_text_arr[:1]))], max_seq_length)
print('train_dataset:',train_dataset)

'''
train_dataset = bert_preprocess.bert_pack_inputs([bert_preprocess.tokenize(tf.constant(train_text_arr))], max_seq_length)
steps_per_epoch = len(train_text_arr) // batch_size
num_train_steps = steps_per_epoch * epochs
# num_warmup_steps = num_train_steps // 10

validation_dataset = bert_preprocess.bert_pack_inputs([bert_preprocess.tokenize(tf.constant(val_text_arr))], max_seq_length)
validation_steps = len(val_text_arr) // batch_size

classifier_model = build_classifier_model(intents_num)

optimizer = optimizer = tf.keras.optimizers.Adam(lr=init_lr)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)

classifier_model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])

classifier_model.fit(train_dataset,train_intents,
  validation_data=(validation_dataset,val_intents),
  steps_per_epoch=steps_per_epoch,
  epochs=epochs,
  batch_size = batch_size,
  validation_steps=validation_steps)

save_folder_path = './saved_models/bert-tiny-L4-H128/'
print('Saving ..')
if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path)
    print('Folder `%s` created' % save_folder_path)
classifier_model.save(save_folder_path)

# test_query =  [['write an email to shabha regarding office tour and cc manisha'],['show me italian restaurants']]
test_query =  ['write an email to shabha regarding office tour and cc manisha']
# test_query = data_text_arr[:5]
test_data = bert_preprocess.bert_pack_inputs([bert_preprocess.tokenize(tf.constant(test_query))], max_seq_length)
# print('Test query prediction:', classifier_model.predict(test_data))
print('Test query prediction:',intents_label_encoder.inverse_transform([np.argmax(classifier_model.predict(test_data))]))
'''
