from torch import nn
import torch

class PositionWiseFeedForward(nn.Module):
    def __init__(self,d_model,hidden,drop_prob=0.1):
        super(PositionWiseFeedForward,self).__init__()
        self.lienar1 = nn.Linear(d_model,hidden)
        self.linear2 = nn.Linear(hidden,d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_prob)

    def forward(self,x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x