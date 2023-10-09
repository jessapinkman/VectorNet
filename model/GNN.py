import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionNet(nn.Module):
    def __init__(self, input_channels=128, key_dim=64, value_dim=64):
        super().__init__()
        self.query = nn.Linear(input_channels, key_dim)
        nn.init.kaiming_normal_(self.query.weight)
        self.key = nn.Linear(input_channels, key_dim)
        nn.init.kaiming_normal_(self.key.weight)
        self.value = nn.Linear(input_channels, value_dim)
        nn.init.kaiming_normal_(self.value.weight)

    def forward(self, polyline_feature):
        Q = F.relu(self.query(polyline_feature))
        K = F.relu(self.key(polyline_feature))
        V = F.relu(self.value(polyline_feature))
        query_res = Q.mm(K.t())
        query_res = query_res / (K.shape[1] ** 0.5)
        attention = F.softmax(input=query_res, dim=1)
        output = attention.mm(V)
        return output + Q