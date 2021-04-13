# -*- coding: utf-8 -*-
"""
@author: rishabbh-sahu
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import tensorflow_hub as hub
import json
import os


class JOINT_TEXT_MODEL():
    def __init__(self, slots_num, intents_num, model_path, learning_rate, name='text_classification'):
        super(JOINT_TEXT_MODEL, self).__init__()

        self.model = None
        self.name = name
        self.num_slot_classes = slots_num
        self.num_seq_classes = intents_num
        self.model_path = model_path
        self.lr = learning_rate

        self.model_params = {
            'num_slot_classes': slots_num,
            'num_sequence_classes': intents_num,
            'model_path': model_path,
            'learning_rate': learning_rate
        }

        print(f'loading the model layer...')
        self.model_layer = hub.KerasLayer(self.model_path, trainable=True, name='bert_layer')
        print(f'model - {self.model_path} successfully loaded..')
        self.build_model()
        self.compile_model()

    def compile_model(self):
        optimizer = tf.keras.optimizers.Adam(lr=float(self.lr))
        losses = {
            'slot_classifier': 'sparse_categorical_crossentropy',
            'sequence_classifier': 'sparse_categorical_crossentropy',
        }
        loss_weights = {'slot_classifier': 3.0, 'sequence_classifier': 1.0}
        metrics = {
            'slot_classifier': 'acc',
            'sequence_classifier': 'acc',
        }
        self.model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics)
        self.model.summary()

    def build_model(self):
        inputs = dict(
            input_word_ids=Input(shape=(None,), dtype=tf.int32, name="input_word_ids"),
            input_mask=Input(shape=(None,), dtype=tf.int32, name="input_mask"),
            input_type_ids=Input(shape=(None,), dtype=tf.int32, name="input_type_ids"),
        )
        pooled_output = self.model_layer(inputs)['pooled_output']
        sequence_output = self.model_layer(inputs)['sequence_output']

        sequence_classifier = Dense(self.num_seq_classes, activation='softmax', name='sequence_classifier')(
            pooled_output)
        slot_classifier = Dense(self.num_slot_classes, activation='softmax', name='slot_classifier')(sequence_output)
        self.model = Model(inputs=inputs, outputs=[slot_classifier, sequence_classifier], name=self.name)

    def fit(self, X, Y, validation_data=None, epochs=5, batch_size=16):
        """
        X: batch of [input_ids, input_mask, segment_ids]
        """
        self.model_params.update({'epochs': epochs, 'batch_size': batch_size})
        history = self.model.fit(X, Y, validation_data=validation_data,
                                 epochs=epochs, batch_size=batch_size, shuffle=False, verbose=2)

    def predict(self, x):
        return self.model.predict(x)

    def save(self, saved_model_path):
        self.model_params.update({'saved_model_path': saved_model_path})
        self.model.save(saved_model_path)
        with open(os.path.join(saved_model_path, 'model_params.json'), 'w') as json_file:
            json.dump(self.model_params, json_file)

