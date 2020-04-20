import torch
from torch import nn
# from ops.Conditional_BN import Conditional_BN
from ops.Adain import Adain

class Conditional_DeConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, upsampl_scale = 2, n_class=2):
        super().__init__()
        self.upsampling = nn.UpsamplingNearest2d(scale_factor=upsampl_scale)
        padding_size = int((kernel_size -1)/2)
        self.same_padding   = nn.ReplicationPad2d(padding_size)
        self.conv           = nn.Conv2d(in_channels = in_channels , out_channels = out_channels , kernel_size= kernel_size, bias= False)
        self.adain          = Adain(out_channels,n_class)
        self.relu           = nn.ReLU()


    def forward(self, input, condition):
        h   = self.upsampling(input)
        h   = self.same_padding(h)
        h   = self.conv(h)
        h   = self.adain(h,condition)
        h   = self.relu(h)
        return h