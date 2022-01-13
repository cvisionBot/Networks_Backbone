from ..layers.convolution import Conv2dBnAct, Conv2dBn
from ..layers.utils import ChannelShuffle
from ..initialize import weight_initialize

import torch
from torch import nn

# Spatial Separable Convolution
class MicroNetStem(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2):
        super(MicroNetStem, self).__init__()
        # out_channels = group (if group > 4 else MaxGroupPoolings)
        gp1, gp2 = self.div(out_channels)
        self.conv1 = Conv2dBn(in_channels=in_channels, out_channels=gp1, kernel_size=(kernel_size, 1), stride=(stride, 1), dilation=1, groups=1)
        self.conv2 = Conv2dBn(in_channels=in_channels, out_channels=gp1 * gp2, kernel_size=(1, kernel_size), stride=(1, stride), dilation=1, groups=gp1)
        self.shuffle = ChannelShuffle(in_channels=gp1)
        self.act = nn.ReLu6()
    
    def forward(self, input):
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.shuffle(output)
        output = self.act(output)
        return output
    
    def div(self, input):
        return input//2, input//2


class _MicroNet_M2(nn.Module):
    def __init__(self, in_channels, classes):
        super(_MicroNet_M2, self).__init__()
        self.micronetStem = MicroNetStem(in_channels, out_channels=8)

    def forward(self, input):
        stem = self.micronetStem(input)
        print('# stem shape : ', stem.shape)


def MicroNet(in_channels, classes=1000, varient=1):
    if varient == 0:
        pass
    elif varient == 1:
        pass
    elif varient == 2:
        model = _MicroNet_M2(in_channels, classes=classes)
    else:
        pass
    weight_initialize(model)
    return model


if __name__ == '__main__':
    model = MicroNet(in_channels=3, classes=1000, varient=2)
    model(torch.rand(1, 3, 224, 224))
