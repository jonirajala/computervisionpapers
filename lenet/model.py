"""
LeNet is a series of convolutional neural network structure

Architecture
- conv
- avgpool
- tanc activation func
- conv
- avgpool
- tanc activation func
- fc
- fc
- fc

"""

from torch import nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv_layer1 = ConvLayer(in_c=1, out_c=6, kernel_size=5, padding=2)
        self.conv_layer2 = ConvLayer(in_c=6, out_c=16, kernel_size=5, padding=0)
        self.linear_block = LinearBlock()

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = x.view(x.size(0), -1)

        y = self.linear_block(x)
        return y

class ConvLayer(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, padding):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, 1, padding)
        self.avg_pool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        x = F.tanh(self.conv(x))
        y = self.avg_pool(x)
        return y

class LinearBlock(nn.Module):
    def __init__(self):
        super(LinearBlock, self).__init__()
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        y = self.fc3(x)
        return y
        