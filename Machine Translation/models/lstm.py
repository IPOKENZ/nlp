import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

class LSTM_Encoder(nn.Module):
    def __init__(self, embedding, hidden_size, num_layers, dropout):
        super(LSTM_Encoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # Create word embedding
        vocab_size, embedding_size = embedding.size()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding.weight.data.copy_(embedding)

        # Create LSTM and linear layers 
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout)

    def forward(self, x, h):
        # Embed text and pass through LSTM
        x = self.embedding(x)
        return self.lstm(x, h)

class LSTM_Decoder(nn.Module):
    def __init__(self, embedding, hidden_size, num_layers, dropout):
        super(LSTM_Decoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # Create word embedding
        vocab_size, embedding_size = embedding.size()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding.weight.data.copy_(embedding)

        # Create LSTM and linear layers 
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, x, h):
        # Embed text and pass through LSTM
        x = self.embedding(x)
        out, h = self.lstm(x, h)
        out = self.linear(out)
        out = self.softmax(out.permute(2,0,1)).permute(1,2,0)
        return out, h
    