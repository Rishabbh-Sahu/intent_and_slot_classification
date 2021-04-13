# -*- coding: utf-8 -*-
"""
@author: rishabbh-sahu
"""

def remove_next_line(txt_arr):
    '''
    data cleansing in order to remove next line symbol may arise while data reading the file

    txt_arr: data cleansing in order to remove next line symbol may arise while data preperation
    return: cleaned text without next line '\n' char
    '''
    return [sub.replace('\n', '') for sub in txt_arr]

def remove_punctuations(txt_arr):
    return txt_arr



