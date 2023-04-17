import torch.nn as nn
import torch.nn.functional as F
import torch
from settings import *

class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)                                     #takes in embedding of input and persistent state, and outputs a single prediction value.

    def forward(self, x, prev_state):
        #print(f"input dimensions: {x.shape}")
        #print(f"prev_state dimensions: {prev_state[0].shape}   {prev_state[1].shape}")
        transformed = torch.unsqueeze(x, -1)
        #print(f"transformed dimensions: {transformed.shape}")

        output, state = self.lstm(transformed, prev_state)
        pred = self.fc(output)
        return pred, state
    
    def init_state(self, sequenceLength):
        h_0 = torch.zeros(num_layers, sequenceLength, hidden_size)       #history and cell initialised as 0.
        c_0 = torch.zeros(num_layers, sequenceLength, hidden_size)
        if torch.cuda.is_available():
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()
        return (h_0, c_0)