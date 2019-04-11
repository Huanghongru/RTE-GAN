import re
import string
import time
import random
import unicodedata
import json_lines as jsonl

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

class phPair:
    """
    A class for premise-hypothesis pair.
    """
    def __init__(self, p, h, l):
        self.premise = p
        self.hypothesis = h
        self.label = l


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def getLabel(anno_labels):
    cnt = {'neutral': 0, 'contradiction': 0, 'entailment': 0}
    res, cur_max = '', 0
    for label in anno_labels:
        if label:
            cnt[label] += 1
            if cnt[label] > cur_max:
                res = label
                cur_max = cnt[label]
    return res

def readData(file_name):
    """
    Read the data. Filter out those neutral data.
    For datum that has multiple labels, select the most one as its label.

    Input: (str) file_name
    Output: (list) premise-hypothesis pairs
    """
    print "Reading data file %s..." % file_name

    ph_pairs = []
    corpus_dict = Lang('en')
    with open(file_name, 'rb') as f:
        for item in jsonl.reader(f):
            p = normalizeString(item['sentence1'])
            h = normalizeString(item['sentence2'])
            l = getLabel(item['annotator_labels'])
            datum = phPair(p, h, l)
            if datum.label != 'neutral':
                ph_pairs.append(datum)
                corpus_dict.addSentence(p)
                corpus_dict.addSentence(h)
    return ph_pairs, corpus_dict

