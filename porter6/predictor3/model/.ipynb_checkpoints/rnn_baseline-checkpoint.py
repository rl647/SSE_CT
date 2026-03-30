import numpy as np
import pandas as pd
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Net(nn.Module):
    def __init__(self, input_size, hidden_size=24, num_layers=2, bidirectional=True, dropout=0.2):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True,
                         bidirectional=bidirectional, nonlinearity='tanh', dropout=dropout)
        self.bidirectional = bidirectional
        if bidirectional:
            self.fc = nn.Linear(hidden_size*2, 1)
        else:
            self.fc = nn.Linear(hidden_size, 1)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = x.transpose(1, 2)
        if self.bidirectional:
            h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        else:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # Forward Prop
        out, hn = self.rnn(x, h0)
        # all training example, last hidden state, all 
        # it is not the last hidden state, it is the last batch
        # print('out.squeeze() ', out.squeeze().size())
        out = self.fc(out)
        out = self.sigmoid(out)
        
        out = out.flatten(start_dim=1)
        return out