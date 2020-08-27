import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base_model import BaseModel

class Baseline_LSTM(BaseModel):
    def __init__(self, output_dim):
        super(Baseline_LSTM, self).__init__()
        
        self.lstm1 = nn.LSTM(input_size= 128, hidden_size= 128, num_layers= 3, batch_first= True)
        self.lstm2 = nn.LSTM(input_size= 128, hidden_size= 128, num_layers= 3, batch_first= True)

        self.fc1 = nn.Linear(in_features= 256, out_features= output_dim)

    def forward(self, x):
        x = torch.squeeze(x)
        x, (hn, cn) = self.lstm1(x)
        x, (hn, cn) = self.lstm2(x)
        x = x.contiguous().view(x.shape[0],-1)
        x = self.fc1(x)
        return x
