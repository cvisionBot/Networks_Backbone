import math
import torch
from torch import nn

class ChannelShuffle(nn.Module):
    def __init__(self, in_channels):
        super(ChannelShuffle, self).__init__()
        self.channels = in_channels

    def forward(self, input):
        b, c, h, w = input.size()
        ch_group = c // self.channels
        output = input.view(b, self.channels, ch_group, h, w)
        output = torch.transpose(output, 1, 2).contiguous()
        output = output.view(b, -1, h, w)
        return output