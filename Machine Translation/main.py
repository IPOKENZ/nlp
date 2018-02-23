from train import train_model, validate_model
from models.lstm import LSTM_Encoder, LSTM_Decoder
from utils import Logger

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

## Data Preprocessing -------------------------------------------------------------------------------------

from torchtext import data
from torchtext import datasets

use_gpu = torch.cuda.is_available()

import spacy
spacy_de = spacy.load('de')
spacy_en = spacy.load('en') #might have to change these, depending on what you named your packages

def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

BOS_WORD = '<s>'
EOS_WORD = '</s>'
DE = data.Field(tokenize=tokenize_de)
EN = data.Field(tokenize=tokenize_en, init_token = BOS_WORD, eos_token = EOS_WORD) # only target needs BOS/EOS

MAX_LEN = 20
train, val, test = datasets.IWSLT.splits(exts=('.de', '.en'), fields=(DE, EN), 
                                         filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and 
                                         len(vars(x)['trg']) <= MAX_LEN)

MIN_FREQ = 5
DE.build_vocab(train.src, min_freq=MIN_FREQ)
EN.build_vocab(train.trg, min_freq=MIN_FREQ)

BATCH_SIZE = 32
train_iter, val_iter = data.BucketIterator.splits((train, val), batch_size=BATCH_SIZE, device=-1,
                                                  repeat=False, sort_key=lambda x: len(x.src))

val_iter_bs1 = data.BucketIterator(val, batch_size=1, device=-1,
                                   repeat=False, sort_key=lambda x: len(x.src))    

## Load in Embeddings ------------------------------------------------------------------------------------------

from torchtext.vocab import Vectors

url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
DE.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url))
EN.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url))

## Build Parameters --------------------------------------------------------------------------------------------

hidden_size = 1000
num_layers = 4
dropout = 0.2
encoder = LSTM_Encoder(DE.vocab.vectors, hidden_size, num_layers, dropout).cuda()
decoder = LSTM_Decoder(EN.vocab.vectors, hidden_size, num_layers, dropout).cuda()
criterion = nn.NLLLoss()
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)

## Train Model -------------------------------------------------------------------------------------------------

logger = Logger()

train_model(train_iter, val_iter_bs1, encoder, decoder, optimizer, criterion, DE, EN,
            max_norm=1.0, num_epochs=50, logger=logger)