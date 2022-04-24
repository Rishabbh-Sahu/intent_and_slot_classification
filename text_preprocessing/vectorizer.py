# -*- coding: utf-8 -*-
"""
@author: rishabbh-sahu
"""

import bert
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizerFast

def label_encoder(unique_classes):
    """
    This routine helps establish the mapping of unique classes to their numeric equivalent
    unique_classes: list [] - unique set of classes to be mapped
    return: model fitted on unique classes to transform where ever required
    """
    le = LabelEncoder()
    le.fit_transform(unique_classes)
    return le

def label_encoder_transform(unique_classes,_label_encoder):
    """
    This routine helps transform unique classes to their numeric equivalent
    unique_classes: list [], unique set of classes to be mapped
    _label_encoder: trained labelEncoder model
    return: transformed numeric values for their corresponding classes
    """
    return _label_encoder.transform(unique_classes).astype(np.int32)

class BERT_PREPROCESSING:

    def __init__(self, model_layer, max_seq_length):
        """
        model_layer: Model layer to be used to create the tokenizer for
        max_seq_length: int - maximum number of tokens to keep in a sequence
        """
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
        """
        param text: Text to tokenize
        param tokenizer: tokenizer used for word splitting
        return: stream of sub-tokens after tokenization
        """
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))

    def get_ids(self, tokens):
        """Token ids from Tokenizer vocab"""
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens, )
        input_ids = token_ids + [0] * (self.max_seq_length - len(token_ids))
        return input_ids

    def get_masks(self, tokens):
        """Mask ids - 1 for valid tokens and 0 for padding"""
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
        """Creating model input for a sequence"""
        stokens = self.tokenizer.tokenize(sentence)
        stokens = stokens[:MAX_LEN]
        stokens = ["[CLS]"] + stokens + ["[SEP]"]
        ids = self.get_ids(stokens)
        masks = self.get_masks(stokens)
        segments = self.get_segments(stokens)
        return ids, masks, segments

    def create_input_array(self, sentences):
        """Creating model input array"""
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
        '''
        create an equivalent tag list corresponds to input ids by assigning first sub-tokens as main tag
        and rest of the sub-tokens as O
        sentence: str - query from seq.in file
        sent_tags: str - corresponding tags from seq.out file
        slot_encoder - slot label encoder to transform class to numeric values
        return: list of transformed tags for the query
        '''
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

# Used Huggingface Fast tokenizer class which can be used to differentiate 
# between main subword vs rest if split of a word happens. Its faster in a 
# sense that, instead of running two for loops (Above), one to iterate over
# sentences and another one to iterate over words (in that sentence), its just 
# uses offset_mapping args to make the job easier for us by just using one for loop.
class BERT_PREPROCESSING_FAST:

    def __init__(self,max_seq_length, bert_model_name = 'bert-base-uncased'):
        '''
        model_layer: Model layer to be used to create the tokenizer for
        max_seq_length: int - maximum number of tokens to keep in a sequence
        '''
        super(BERT_PREPROCESSING_FAST,self).__init__()
        self.max_seq_length = max_seq_length
        self.fastTokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    def tokenize_text(self,text):
        '''
        param text: Text to tokenize
        param tokenizer: tokenizer used for word splitting
        return: stream of sub-tokens after tokenization
        '''
        return self.fastTokenizer.convert_tokens_to_ids(self.fastTokenizer.tokenize(text))

    def create_model_input(self,encodings):
        '''create numpy array as the model input from the transformer's encoded object'''
        return {'input_word_ids': np.array(encodings.input_ids),
                'input_mask': np.array(encodings.attention_mask),
                'input_type_ids': np.array(encodings.token_type_ids),
                }

    def encode_tags(self, sent_tags: str, encodings, slot_encoder):
        '''
        Ref - https://huggingface.co/transformers/master/custom_datasets.html#token-classification-with-w-nut-emerging-entities
        create an equivalent tag list corresponds to input ids by assigning first sub-tokens as main tag
        and rest of the sub-tokens as O
        sent_tags: str - corresponding tags from seq.out file
        encodings: fast tokenizer object - contains all the necessary inputs array for the sentences/query
        slot_encoder - slot label encoder to transform class to numeric values
        return: list of transformed tags for the query
        '''
        labels = [slot_encoder.transform(doc) for doc in sent_tags]
        encoded_labels = []
        for doc_labels, doc_offset, am in tqdm(zip(labels, encodings.offset_mapping, encodings.attention_mask),
                                               position=0, leave=True):
            # create an empty array of -100
            doc_enc_labels = am * slot_encoder.transform(['O'])
            arr_offset = np.array(doc_offset)
            # set labels whose first offset position is 0 and the second is not 0
            main_tokens_flags = (arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)
            doc_enc_labels[main_tokens_flags] = doc_labels[:sum(main_tokens_flags)]
            encoded_labels.append(doc_enc_labels.tolist())
        return encoded_labels
