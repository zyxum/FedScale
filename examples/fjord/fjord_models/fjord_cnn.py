import torch
import torch.nn as nn


class FjORD_CNN(nn.Module):
    """
    conv - relu - maxpool - conv - relu - maxpool - linear1 - linear2
    """


    def __init__(self, num_classes: int=10):
        super(FjORD_CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding="same")
        self.activation1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding="same")
        self.activation2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear1 = nn.Linear(in_features=64*7*7, out_features=2048)
        self.linear2 = nn.Linear(in_features=2048, out_features=num_classes)


    def forward(self, x):
        x = self.pool1(self.activation1(self.conv1(x)))
        x = self.pool2(self.activation2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.linear2(self.linear1(x))
        x = nn.functional.softmax(x, dim=1)
        return x

