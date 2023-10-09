import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchsnooper


class Subgraph_Layer(nn.Module):
    def __init__(self, input_channels=128, hidden_channels=32, out_channels=64):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, out_channels)
        )
        # single fully connected network
        # nn.init.kaiming_normal_(tensor=self.mlp.weight) # init mlp weights

    # @torchsnooper.snoop()
    def forward(self, input): # (18,4)
        hidden = self.mlp(input).unsqueeze(0)                              # Vi, 一个全连接层,unsqueeze增加一维 torch.Size([r, c]) -> torch.Size([1, r, c])
        encode_data = F.relu(F.layer_norm(hidden, hidden.shape[1:]))      # layer norm and relu
        kernel_size = encode_data.shape[1]                                 # 18
        maxpool = nn.MaxPool1d(kernel_size)                                # maxpool
        polyline_feature = maxpool(encode_data.transpose(1,2)).squeeze()   # Vj, transpose and squeeze
        polyline_feature = polyline_feature.repeat(kernel_size, 1)
        output = torch.cat([encode_data.squeeze(), polyline_feature], 1)   # concanate->(r, 2*c)
        return output   #(30, 128)


class SubgraphNet(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.sublayer1 = Subgraph_Layer(input_channels)
        self.sublayer2 = Subgraph_Layer()
        self.sublayer3 = Subgraph_Layer()  # output = 128


    def forward(self, input):                # torch.shape([18, 4])
        output1 = self.sublayer1(input)      # 调用SubgraphNet_Layer.forward(input)，  out -> (30, 128)
        output2 = self.sublayer2(output1)    # (128, 128)
        output3 = self.sublayer3(output2)    # (128, 128)
        kernel_size = output3.shape[0]       # 128
        maxpool = nn.MaxPool1d(kernel_size)
        polyline_feature = maxpool(output3.unsqueeze(1).transpose(0,2)).squeeze()
        return polyline_feature              # (128)