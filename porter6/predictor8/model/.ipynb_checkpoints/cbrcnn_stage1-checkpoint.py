import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.input_size = input_size
        
        ## CNN1D
        #  Input: (N, C_{in}, L_{in}) - (batch_size, num_input_channel, len_sequence)
        # Output: (N, C_{out}, L_{out} - ()
        
        ## LSTM/RNN
        #  Input: (N,L,H_{in}) - (batch_size, len_seq, input_size)
        # Output: (N,L,D∗H_{out}) - (batch_size, len_seq, 2*hidden_size), D=2 if bidirectional, otherwise, 1
        self.rnn1 = nn.RNN(input_size=self.input_size, hidden_size=40, num_layers=2, batch_first=True, bidirectional=True)
        self.conv11 = nn.Conv1d(in_channels=self.rnn1.hidden_size*2, out_channels=10, kernel_size=7, stride=1, padding=3)
        self.tanh = nn.Tanh()
        self.conv12 = nn.Conv1d(in_channels=self.conv11.out_channels, out_channels=9, kernel_size=1, stride=1, padding=0) # out_channels should be the number of classes.
        
    def forward(self, x):

        x = x.transpose(1, 2)
        
        # initialize h0
        # bidirectional
        h0 = torch.zeros(self.rnn1.num_layers*2, x.size(0), self.rnn1.hidden_size).to(device)
        # Forward process
        x, hn = self.rnn1(x, h0)
        x = self.conv11(x.transpose(1, 2))
        x = self.tanh(x)
        x = self.conv12(x)
        out = F.softmax(torch.transpose(x, 1, 2), dim=2)
        #print(out.size())
        return out