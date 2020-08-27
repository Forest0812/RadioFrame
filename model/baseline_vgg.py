import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base_model import BaseModel


class Baseline_VGG(BaseModel):
    def __init__(self, output_dim):
        super(Baseline_VGG, self).__init__()

        # input(batch, 2, 128)
        self.conv1 = nn.Sequential(
            nn.BatchNorm1d(2),
            nn.Conv1d(in_channels = 2, out_channels= 64, kernel_size= 3, stride= 1, padding= 1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        # after conv1(batch, 64, 64)
        self.conv2 = nn.Sequential(
            nn.BatchNorm1d(64),
            nn.Conv1d(in_channels = 64, out_channels= 64, kernel_size= 3, stride= 1, padding= 1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        # after conv2(batch, 64, 32)
        self.conv3 = nn.Sequential(
            nn.BatchNorm1d(64),
            nn.Conv1d(in_channels = 64, out_channels= 64, kernel_size= 3, stride= 1, padding= 1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        # after conv3(batch, 64, 16)
        self.conv4 = nn.Sequential(
            nn.BatchNorm1d(64),
            nn.Conv1d(in_channels = 64, out_channels= 64, kernel_size= 3, stride= 1, padding= 1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        # after conv4(batch, 64, 8)
        self.fc1 = nn.Sequential(
            nn.Linear(64 * 8, 128),
            nn.SELU(),
            # nn.Dropout(0.5),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.SELU(),
            # nn.Dropout(0.5),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        x = torch.squeeze(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    