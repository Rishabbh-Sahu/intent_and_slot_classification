# -*- coding: utf-8 -*-
"""
@author: rishabbh-sahu
"""

import bert
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tqdm import tqdm

def label_encoder(unique_classes):
    le = LabelEncoder()
    le.fit_transform(unique_classes)
    return le

def label_encoder_transform(unique_classes,_label_encoder):
    return _label_encoder.transform(unique_classes).astype(np.int32)

class BERT_PREPROCESSING:

    def __init__(self, model_layer, max_seq_length):
        '''
        model_layer: Model layer to be used to create the tokenizer for
        return: tokenizer compatible with the model i.e. bert, albert etc.
        '''
        super(BERT_PREPROCESSING,self).__init__()
        self.model_layer = model_layer
        self.max_seq_length = max_seq_length
        FullTokenizer = bert.bert_tokenization.FullTokenizer
        vocab_file = self.model_layer.resolved_object.vocab_file.asset_path.numpy()
        do_lower_case = self.model_layer.resolved_object.do_lower_case.numpy()
        self.tokenizer = FullTokenizer(vocab_file, do_lower_case)
        print('vocabulary_file:', type(vocab_file), '\nto_lower_case:', type(do_lower_case))
        print('tokenizer.vocab:', len(self.tokenizer.vocab))

    def tokenize_text(self,text):
        '''
        param text: Text to tokenize
        param tokenizer: tokenizer used for word splitting
        return: stream of sub-tokens after tokenization
        '''
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))

    def get_ids(self, tokens):
        """Token ids from Tokenizer vocab"""
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens, )
        input_ids = token_ids + [0] * (self.max_seq_length - len(token_ids))
        return input_ids

    def get_masks(self, tokens):
        return [1] * len(tokens) + [0] * (self.max_seq_length - len(tokens))

    def get_segments(self, tokens):
        """Segments: 0 for the first sequence, 1 for the second"""
        segments = []
        current_segment_id = 0
        for token in tokens:
            segments.append(current_segment_id)
            if token == "[SEP]":
                current_segment_id = 1
        return segments + [0] * (self.max_seq_length - len(tokens))

    def create_single_input(self, sentence, MAX_LEN):
        stokens = self.tokenizer.tokenize(sentence)
        stokens = stokens[:MAX_LEN]
        stokens = ["[CLS]"] + stokens + ["[SEP]"]
        ids = self.get_ids(stokens)
        masks = self.get_masks(stokens)
        segments = self.get_segments(stokens)
        return ids, masks, segments

    def create_input_array(self, sentences):
        input_ids, input_masks, input_segments = [], [], []
        for sentence in tqdm(sentences, position=0, leave=True):
            ids, masks, segments = self.create_single_input(sentence, self.max_seq_length - 2)
            input_ids.append(ids)
            input_masks.append(masks)
            input_segments.append(segments)
        return {'input_word_ids': np.array(input_ids),
                'input_mask': np.array(input_masks),
                'input_type_ids': np.array(input_segments), }

    def get_tag_labels(self, sentence: str, sent_tags: str, slot_encoder):
        words = sentence.split()  # whitespace tokenizer
        tags = sent_tags.split()
        tags_extended = []
        for i, word in enumerate(words):
            tokens = self.tokenizer.tokenize(word)
            tags_extended.append(tags[i])
            if len(tokens) > 1:
                tags_extended.extend((len(tokens) - 1) * ['O'])
        tags_extended = slot_encoder.transform(tags_extended)
        # [CLS] token takes 'O' at the start - BERT INPUT
        tags_extended = np.insert(tags_extended, 0, slot_encoder.transform(['O']), axis=0)
        # [SEP] token takes 'O' at the end - BERT INPUT
        tags_extended = np.append(tags_extended, slot_encoder.transform(['O']))
        # Insert PAD's if max seq length > lenght of tags
        if len(tags_extended) < self.max_seq_length:
            tags_extended = np.append(tags_extended,slot_encoder.transform(['<PAD>'] * (self.max_seq_length - len(tags_extended))))
        else:
            tags_extended = tags_extended[:self.max_seq_length]
        return tags_extended
