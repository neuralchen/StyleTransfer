import torch
from torch import nn
# from ops.Conditional_BN import Conditional_BN
from ops.Adain import Adain

class Conditional_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, n_class = 2, stride=1):
        super().__init__()
        padding_size = int((kernel_size -1)/2)
        self.pad    = nn.ReplicationPad2d(padding_size)
        self.conv   = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, stride=stride, kernel_size= kernel_size, bias= False)
        self.adain  = Adain(out_channels, n_class)
        self.relu   = nn.ReLU()

    def forward(self, input, condition):
        h   = self.pad(input)
        h   = self.conv(h)
        h   = self.adain(h, condition)
        h   = self.relu(h)
        return h