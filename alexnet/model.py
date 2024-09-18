"""
This landmark paper introduced AlexNet, which achieved unprecedented performance on the ImageNet challenge and popularized the use of deep convolutional neural networks. It also utilized GPUs for training large-scale neural networks.

model:
- eight learned layers
    - five convolutional
    - three fully-connected
- ReLU Nonlinearity
- LocalResponseNorm after the ReLU activation in certain layers.
    - Specifically, it takes the output of each neuron and divides it by a term that depends on the outputs of neighboring neurons.
    - This term is calculated using the sum of squared outputs of nearby neurons within the same layer.
- MaxPooling
- Dropout
    - We use dropout in the first two fully-connected layers of Figure

Training:
- stochastic gradient descent
- batch size of 128 examples
- momentum of 0.9
- weight decay of 0.0005 
"""

from torch import nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.conv5 = nn.Conv2d(
            in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1
        )
    
        self.lr_norm1 = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=1, k=2)
        self.lr_norm2 = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=1, k=2)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.fc1 = nn.Linear(in_features=512, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=10)
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.lr_norm1(x)
        x = self.max_pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.lr_norm2(x)
        x = self.max_pool2(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = F.relu(x)

        x = self.conv5(x)
        x = F.relu(x)

        x = self.max_pool3(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)

        return x
