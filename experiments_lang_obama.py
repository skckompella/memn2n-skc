
import numpy as np
import re
import os


PATH_TO_DATA = '/Users/skc/Projects/memn2n-lang-keras/data'
TRAIN_FILE = PATH_TO_DATA + 'input.txt'

def tokenize(sentence):
    sentence = sentence.lower()
    return re.findall("[\'\w\d\-\*]+|[^a-zA-Z\d\s]+", sentence)

def parse_speeches(lines):
    count = 0
    data = []
    speeches = []
    for line in lines:
        line = line.decode('utf-8').strip()
        data.append(tokenize(line))
        count+=1 if line == '' else 0
        if count > 1:
            speeches.append(data)


    return data

def get_speeches():

    with open(TRAIN_FILE , 'r') as f:
        lines = f.readlines()
    data = parse_speeches(lines)


def vectorize_text():

    pass