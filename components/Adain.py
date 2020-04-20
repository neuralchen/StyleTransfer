import torch
from torch import nn

class Adain(nn.Module):
    def __init__(self, in_channel, n_class=6):
        super().__init__()

        self.instance_norm = nn.InstanceNorm2d(in_channel,momentum=0, affine=False)

        self.embed = nn.Linear(n_class * 16, in_channel* 2)
        self.embed.weight.data[:, :in_channel] = 1
        self.embed.weight.data[:, in_channel:] = 0

    def forward(self, input, class_id):
        out = self.instance_norm(input)
        embed = self.embed(class_id)
        sigma, mu = embed.chunk(2, 1)
        sigma = sigma.unsqueeze(2).unsqueeze(3)
        mu = mu.unsqueeze(2).unsqueeze(3)
        out = sigma * out + mu

        return out