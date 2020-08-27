import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base_model import BaseModel


class VTCNN2(BaseModel):
    def __init__(self):
        super(VTCNN2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ZeroPad2d(padding = (2, 2)),
            nn.Conv2d(in_channels = 1, out_channels = 256, kernel_size = (1, 3)),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ZeroPad2d(padding = (2, 2)),
            nn.Conv2d(in_channels = 256, out_channels = 80, kernel_size=(2, 3)),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(10560, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 11)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.shape[0],-1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    