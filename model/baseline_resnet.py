import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base_model import BaseModel

class Res_Unit(nn.Module):
    def __init__(self, input_dim):
        super(Res_Unit, self).__init__()
        self.conv1 = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Conv1d(in_channels= input_dim, out_channels= input_dim, kernel_size= 3, padding= 1, stride= 1),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Conv1d(in_channels= input_dim, out_channels= input_dim, kernel_size= 3, padding= 1, stride= 1),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(x)
        output = output + x
        output = self.relu(output)
        return output

class Res_Stack(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Res_Stack, self).__init__()
        self.conv1x1 = nn.Conv1d(in_channels= input_dim, out_channels= output_dim, kernel_size= 1, padding= 0, stride= 1)
        self.res_unit1 = Res_Unit(output_dim)
        self.res_unit2 = Res_Unit(output_dim)
        self.max_pooling = nn.MaxPool1d(kernel_size= 2)
    
    def forward(self, x):
        x = self.conv1x1(x)
        x = self.res_unit1(x)
        x = self.res_unit2(x)
        x = self.max_pooling(x)
        return x


class Baseline_ResNet(BaseModel):
    def __init__(self, output_dim):
        super(Baseline_ResNet, self).__init__()
        # after res_stack1 (batch, 32, 64)
        self.res_stack1 = Res_Stack(input_dim= 2, output_dim= 32)
        # after res_stack1 (batch, 32, 32)
        self.res_stack2 = Res_Stack(input_dim= 32, output_dim= 32)
        # after res_stack1 (batch, 32, 16)
        self.res_stack3 = Res_Stack(input_dim= 32, output_dim= 32)
        # after res_stack1 (batch, 32, 8)
        self.res_stack4 = Res_Stack(input_dim= 32, output_dim= 32)
        
        self.fc1 = nn.Sequential(
            nn.Linear(32*8, 128),
            nn.SELU(),
            nn.AlphaDropout(0.5),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.SELU(),
            nn.AlphaDropout(0.5),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(64, output_dim),
        )


    def forward(self, x):
        x = torch.squeeze(x)
        x = self.res_stack1(x)
        x = self.res_stack2(x)
        x = self.res_stack3(x)
        x = self.res_stack4(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
