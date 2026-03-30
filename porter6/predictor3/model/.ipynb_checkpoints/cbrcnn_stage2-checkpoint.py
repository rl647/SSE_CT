import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Net(nn.Module):
    def __init__(self, input_size=3):
        super(Net, self).__init__()
        self.input_size = input_size
        
        ## CNN1D
        #  Input: (N, C_{in}, L_{in}) - (batch_size, num_input_channel, len_sequence)
        # Output: (N, C_{out}, L_{out} - ()
        
        ## LSTM/RNN
        #  Input: (N,L,H_{in}) - (batch_size, len_seq, input_size)
        # Output: (N,L,D∗H_{out}) - (batch_size, len_seq, 2*hidden_size), D=2 if bidirectional, otherwise, 1
        self.conv20 = nn.Conv1d(in_channels=self.input_size, out_channels=15, kernel_size=21, stride=1, padding=10)
        self.rnn2 = nn.RNN(input_size=self.conv20.out_channels, hidden_size=20, num_layers=2, batch_first=True, bidirectional=True)
        self.conv21 = nn.Conv1d(in_channels=self.rnn2.hidden_size*2, out_channels=5, kernel_size=7, stride=1, padding=3)
        self.tanh = nn.Tanh()
        self.conv22 = nn.Conv1d(in_channels=self.conv21.out_channels, out_channels=9, kernel_size=1, stride=1, padding=0) # out_channels should be the number of classes.
        
    def forward(self, x):

        x = self.conv20(x)
        x = self.tanh(x)
        
        x = x.transpose(1, 2)
        
        # initialize h0
        # bidirectional
        h0 = torch.zeros(self.rnn2.num_layers*2, x.size(0), self.rnn2.hidden_size).to(device)
        # Forward process
        x, hn = self.rnn2(x, h0)
        x = self.conv21(x.transpose(1, 2))
        x = self.tanh(x)
        x = self.conv22(x)
        out = F.softmax(torch.transpose(x, 1, 2), dim=2)
        return out