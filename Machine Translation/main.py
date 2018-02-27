from train import train_model, validate_model
from models.lstm import LSTM_Encoder, LSTM_Decoder, LSTM_Attention_Decoder
from utils import Logger
import argparse

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
DE.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url)) #Not the right embedding but gives us the right sizes (FIX LATER)
EN.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url))

## Build Parameters --------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Language Model')
parser.add_argument('--pretrain', default=1, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--dropout', default=0.2, type=float)
parser.add_argument('--num_epochs', default=5, type=int)
parser.add_argument('--beam_width', default=10, type=int)

args = parser.parse_args()

hidden_size = 300
num_layers = 4
dropout = args.dropout
encoder = LSTM_Encoder(DE.vocab.vectors, hidden_size, num_layers, args.dropout).cuda()
# decoder = LSTM_Decoder(EN.vocab.vectors, hidden_size, num_layers, dropout).cuda()
attn_decoder = LSTM_Attention_Decoder(EN.vocab.vectors, hidden_size, num_layers, args.dropout).cuda()
criterion = nn.NLLLoss()
optimizer = optim.Adam(list(filter(lambda x: x.requires_grad, encoder.parameters())) + 
                       list(filter(lambda x: x.requires_grad, attn_decoder.parameters())), lr=args.lr)

## Train Model -------------------------------------------------------------------------------------------------

if args.pretrain:
	attn_decoder.load_state_dict(torch.load('saves/decoder.pkl'))
	encoder.load_state_dict(torch.load('saves/encoder.pkl'))
	print('Loaded pretrained model.')

logger = Logger()
 
if use_gpu: 
	print("CUDA is available, hooray!")

train_model(train_iter, val_iter, val_iter_bs1, encoder, attn_decoder, optimizer, criterion, DE, EN,
            max_norm=1.0, num_epochs=args.num_epochs, logger=logger, beam_width=args.beam_width)	

