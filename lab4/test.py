# importing block

from __future__ import unicode_literals, print_function, division
from io import open
import string
import re
import json
import random
import time
import math
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
from os import system
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

torch.manual_seed(100)


# Experimental Setup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1
padding_token = 2
MAX_LENGTH = 100
#----------Hyper Parameters----------#
hidden_size = 256
#The number of vocabulary
vocab_size = 29
teacher_forcing_ratio = 0.6
LR = 0.05

# read in json files

with open('train.json', 'r') as data:
    train_data = json.loads(data.read())
    
with open('test.json', 'r') as data:
    test_data = json.loads(data.read())
    
vocab = dict()
vocab['SOS'] = 0
vocab['EOS'] = 1
vocab['PAD'] = 2
for i, alphabet in enumerate(string.ascii_lowercase):
    vocab[alphabet] = i + 3
    


# return tensors of shape [seq_len, batch_size]
def make_tensors(words):
    # words: list of tokens
    res_tensor = list()
    
    # 1. translate to indexes & add EOS
    for word in words:
        word = [vocab[alphabet] for alphabet in word]
        word.append(vocab['EOS'])
        res_tensor.append(np.array(word))
        
#     # 2. find max_len and record lens
#     lens = [len(word) for word in res_tensor]
#     max_len = max(lens)
    
#     # 3. padding
#     for word in res_tensor:
#         while len(word) < max_len:
#             word.append(vocab['PAD'])
    
    # 4. adjust shape and cast to tensor
    res_tensor = np.array(res_tensor)
    res_tensor = res_tensor.T
    print(res_tensor)
    print(res_tensor.dtype)
#     print(res_tensor[0].dtype)
    res_tensor = torch.from_numpy(res_tensor).type(torch.LongTensor)
    
    if torch.cuda.is_available():
        res_tensor = res_tensor.to(device)
    
    
    return res_tensor
    

def create_pairs(samples):
    
    pairs = list()
    for pair in samples:
        inputs = pair['input']
        target = [pair['target']]
        batch_size = len(inputs)
        
        inputs = make_tensors(inputs)
#         target = make_tensors(target)
        for i in range(batch_size):
            pairs.append((inputs, target))
        
    return pairs

print(create_pairs(train_data))