
import numpy as np
import re
import os


PATH_TO_DATA = '/Users/skc/Projects/memn2n-lang-keras/data'
TRAIN_FILE = PATH_TO_DATA + 'input.txt'

def tokenize(sentence):
    sentence = sentence.lower()
    return re.findall("[\'\w\d\-\*]+|[^a-zA-Z\d\s]+", sentence)

def parse_speeches(lines)#
    count = 0
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        count+=1 if line == '' else count=0
        print count

def get_speeches():

    with open(TRAIN_FILE , 'r') as f:
        lines = f.readlines()
