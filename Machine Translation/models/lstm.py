import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

use_gpu = torch.cuda.is_available()

class LSTM_Encoder(nn.Module):
    def __init__(self, embedding, hidden_size, num_layers, dropout, bidirectional=True):
        super(LSTM_Encoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        # Create word embedding
        vocab_size, embedding_size = embedding.size()
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        # Create LSTM and linear layers 
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Build initial hidden state
        batch_size = x.size(1)
        num_layers = self.num_layers * 2 if self.bidirectional else self.num_layers
        init = Variable(torch.zeros(num_layers, batch_size, self.hidden_size))
        init = init.cuda() if use_gpu else init
        h0 = (init, init.clone())

        # Embed text and pass through LSTM
        x = self.embedding(x)
        x = self.dropout(x)
        out, h = self.lstm(x, h0)
        
        # If it's bidirectional, we need to add the outputs together
        out = out.view(out.size(0), out.size(1), 2, -1).sum(2)
        
        return out, h

class LSTM_Decoder(nn.Module):
    def __init__(self, embedding, hidden_size, num_layers, dropout):
        super(LSTM_Decoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # Create word embedding
        vocab_size, embedding_size = embedding.size()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding.weight.data.copy_(embedding)
        self.embedding.weight.requires_grad=False

        # Create LSTM and linear layers 
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, h, encoder_outputs):
        # Embed text and pass through LSTM
        x = self.embedding(x)
        x = self.dropout(x)
        out, h = self.lstm(x, h)
        out = self.linear(out)
        out = self.softmax(out.permute(2,0,1)).permute(1,2,0)
        return out, h

class LSTM_Attention_Decoder(nn.Module):
    def __init__(self, embedding, hidden_size, num_layers, dropout):
        super(LSTM_Attention_Decoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # Create word embedding
        vocab_size, embedding_size = embedding.size()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding.weight.data.copy_(embedding)
        self.embedding.weight.requires_grad=False
        
        # Create Attention Layers
        # self.attn = nn.Bilinear(self.hidden_size, self.hidden_size, self.max_length)
        # self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
 
        # Create LSTM and linear layers 
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax()
        self.dropout = nn.Dropout(dropout)
        
        # Weight-tie the FC layer to our embeddings
        # self.linear.params = self.embedding.weight

    def forward(self, x, h, encoder_outputs):
        # Embed text and pass through LSTM
        x = self.embedding(x)
        x = self.dropout(x)
        out, h = self.lstm(x, h)
                    
        # Now apply attention
        attn_weights = torch.bmm(encoder_outputs.permute(1,0,2), out.permute(1,2,0))
        attn_weights = F.softmax(attn_weights.permute(0, 2, 1))
        attn_applied = torch.bmm(attn_weights, encoder_outputs.permute(1,0,2))
        
        # out = torch.cat((out, attn_applied.permute(1,0,2)), 2) <--- Need to double FC layer for this one  
        out = torch.add(out, attn_applied.permute(1,0,2))
        out = self.linear(out)
        out = self.softmax(out.permute(2,0,1)).permute(1,2,0)
        return out, h
        
