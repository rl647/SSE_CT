import numpy as np
import pandas as pd
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Net(nn.Module):
    def __init__(self, input_size, hidden_size=24, num_layers=2, bidirectional=True, dropout=0.25):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                         bidirectional=bidirectional)
        self.bidirectional = bidirectional
        if bidirectional:
            self.fc = nn.Linear(hidden_size*2, 1)
        else:
            self.fc = nn.Linear(hidden_size, 1)
        
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.transpose(1, 2)
        if self.bidirectional:
            h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        else:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            
        # Forward Prop
        out, (hn, cn) = self.lstm(x,  (h0, c0))
        
        # all training example, last hidden state, all 
        # it is not last hidden state, it is the last batch
        # print('out.squeeze() ', out.squeeze().size())
        out = self.dropout(out)
        out = self.fc(out)
        out = self.sigmoid(out)
        
        out = out.flatten(start_dim=1)
        return out