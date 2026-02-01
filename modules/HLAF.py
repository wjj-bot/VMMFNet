import torch
import torch.nn as nn

class ChannelAttention_HLAF(nn.Module):
    def __init__(self, in_planes, ratio=4, flag=True):
        super(ChannelAttention_HLAF, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.conv1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.flag = flag
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.conv2(self.relu(self.conv1(self.avg_pool(x))))
        max_out = self.conv2(self.relu(self.conv1(self.max_pool(x))))
        out = avg_out + max_out
        res = self.sigmoid(out)
        return res * x if self.flag else res

class Multiply(nn.Module):
    def __init__(self):
        super(Multiply, self).__init__()

    def forward(self, x):
        return x[0] * x[1]

class HLAF(nn.Module):
    def __init__(self, c1, c2):
        super(HLAF, self).__init__()
        self.ca = ChannelAttention_HLAF(c1, ratio=4, flag=False)
        self.mul = Multiply()

    def forward(self, x):
        # x[0]: high-level semantic feature
        # x[1]: low-level spatial feature
        att = self.ca(x[0])
        fused = self.mul([att, x[1]])
        return fused

class Add_HLAF(nn.Module):
    def __init__(self):
        super(Add_HLAF, self).__init__()

    def forward(self, x):
        # Direct summation is more memory-efficient than stack+sum
        res = x[0]
        for i in range(1, len(x)):
            res = res + x[i]
        return res