#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:


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
teacher_forcing_ratio = 0.66
LR = 0.05


# In[4]:


# BLEU-4 validation function

def compute_bleu(output, reference):
    cc = SmoothingFunction()
    if len(reference) == 3:
        weights = (0.33,0.33,0.33)
    else:
        weights = (0.25,0.25,0.25,0.25)
    return sentence_bleu([reference], output,weights=weights,smoothing_function=cc.method1)


# In[5]:


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
        word_encode = [vocab[alphabet] for alphabet in word]
        word_encode.append(vocab['EOS'])
        res_tensor.append(word_encode)
        
        
#     2. find max_len and record lens
#     lens = [len(word) for word in res_tensor]
#     max_len = max(lens)
    
#     3. padding
#     for word in res_tensor:
#         while len(word) < max_len:
#             word.append(vocab['PAD'])
    
    # 4. adjust shape and cast to tensor
    res_tensor = np.array(res_tensor)
    res_tensor = res_tensor.T
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
        
        target = make_tensors(target)
        for i in inputs:
            input_tensor = make_tensors([i])
            pairs.append((input_tensor, target))
        
    return pairs


# In[6]:


# Encoder & Decoder implementation

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output, hidden = self.lstm(embedded, hidden)
        
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        
        output, hidden = self.lstm(output, hidden)
        
        output = torch.reshape(self.out(output), (1, -1))
        
        
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# In[7]:


# training function

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = (encoder.initHidden(), encoder.initHidden())

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    loss = 0

    #----------sequence to sequence part for encoder----------#
    for i in range(input_length):
        encoder_output, encoder_hidden = encoder(torch.unsqueeze(input_tensor[i], 0), encoder_hidden)
        
#     print(encoder_hidden[0].shape, encoder_hidden[1].shape)

    decoder_input = torch.tensor([[SOS_token]], device=device)
#     print('decoder input', decoder_input)

    decoder_hidden = encoder_hidden
#     print('decoder state', decoder_hidden[0].shape, decoder_hidden[1].shape)

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False


    #----------sequence to sequence part for decoder----------#
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di, :]) # input shape: batch_sizeXC, C=class_num, target shape: batch_size
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


# In[8]:


# time counting functions

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


# In[9]:


# training entry

def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    
    # your own dataloader
    training_pairs = create_pairs(train_data)

    criterion = nn.CrossEntropyLoss()
    
    bleu_scores = list()
    train_loss = list()

    for iter in range(1, n_iters + 1):
        print('='*40, f'[Epoch {iter} starts]', '='*40)
        encoder.train()
        decoder.train()
        for case in range(len(training_pairs)):
            training_pair = training_pairs[case - 1]
            input_tensor = training_pair[0].to(device) # shape seq_len X 1
            target_tensor = training_pair[1].to(device) # shape seq_len X 1

            loss = train(input_tensor, target_tensor, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion, input_tensor.size(0))
            print_loss_total += loss
            plot_loss_total += loss           
            
            
            if case % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, (case + 1) / len(training_pairs)),
                                             case, case / len(training_pairs) * 100, print_loss_avg))
        
        print('\n')
        encoder.eval()
        decoder.eval()
        
        
        bleu_scores.append(evaluate(encoder, decoder, test_data, False))
        train_loss.append(plot_loss_total / len(training_pairs))
        
        plot_loss_total = 0
        
    plot_curve(train_loss, n_iters, 'Training Loss', 'Loss')
    plot_curve(bleu_scores, n_iters, 'BLEU Scores', 'BLEU')


# In[10]:


def translate(dictionary, code):
    res = ''
    for c in code:
        for key, val in dictionary.items():
            if val == c and c > 2:
                res += key
    return res


# In[11]:


def evaluate(encoder, decoder, test_data, print_info=True):
    with torch.no_grad():
        test_pair = create_pairs(test_data)
        scores = list()
        for data, label in test_pair:
            encoder_hidden = (encoder.initHidden(), encoder.initHidden())
            
            encoder_output, hidden_state = encoder(data, encoder_hidden)
            
            decoder_input = torch.tensor([[SOS_token]], device=device)
            
            decoder_hidden = hidden_state
            
            decoder_outputs = list()
            
            for d in range(100):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                
                topv, topi = decoder_output.data.topk(1)
                if topi.item() == EOS_token:
                    break
                else:
                    decoder_outputs.append(translate(vocab, [topi.item()]))
                decoder_input = topi.squeeze().detach()
            
            input_case = translate(vocab, data.T[0].tolist())
            label = translate(vocab, label.T[0].tolist())
            output = "".join(decoder_outputs)
            score = compute_bleu(output, label)
            scores.append(score)
            
            if print_info:
                print('='*40)
                print('Input: ', input_case)
                print('Label: ', label)
                print('Model Output: ', output)
                print(f'Scores: {score}\n')
    print('='*40, '\n')
    score_avg = sum(scores)/len(scores)
    print('Average Score: ', score_avg)
    
    return score_avg


# In[12]:


def plot_curve(content, n_iters, y_label, fig_name):
    fig = plt.figure(figsize=(12, 8), dpi=300)
    plt.title(f'{fig_name}')
    plt.xlabel('Epoch')
    plt.ylabel(y_label)
    
    plt.plot(content, linewidth=0.8)
        
    plt.savefig(f'{fig_name}.jpg')
    plt.show()
    
    return
    


# In[14]:


# initialize model
testing = True
encoder1 = EncoderRNN(vocab_size, hidden_size).to(device)
decoder1 = DecoderRNN(hidden_size, vocab_size).to(device)

if testing:
    encoder1.load_state_dict(torch.load('encoder_best.pth'))
    decoder1.load_state_dict(torch.load('decoder_best.pth'))

print(encoder1)
print(decoder1)


# In[13]:


trainIters(encoder1, decoder1, 75, print_every=1000)


# In[15]:


evaluate(encoder1, decoder1, test_data)


# In[15]:


with open('new_test.json', 'r') as data:
    newtest_data = json.loads(data.read())


# In[16]:


evaluate(encoder1, decoder1, newtest_data)


# In[17]:


torch.save(encoder1.state_dict(), 'encoder_best.pth')
torch.save(decoder1.state_dict(), 'decoder_best.pth')

