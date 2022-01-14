from ..layers.convolution import Conv2dBnAct, SepConvBnAct, DepthSepConvBnAct
from ..initialize import weight_initialize

import torch
from torch import nn

# Spatial Separable Convolution
class MicroNetStem(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2):
        super(MicroNetStem, self).__init__()
        self.conv = SepConvBnAct(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2)
    
    def forward(self, input):
        output = self.conv(input)
        return output


class Make_Layers(nn.Module):
    def __init__(self, layers_configs):
        super(Make_Layers, self).__init__()
        self.layers_configs = layers.configs
        self.layer = self.microBlock(self.layers_configs)

    def forward(self, input):
        return self.layer(input)

    def microBlock(self, layers_configs):
        layers = []
        


class MicroBlockA(nn.Module):
    def __init__(self, in_channels, out_channels, expand, kernel_size, stride, act=None):
        super(MicroBlockA, self).__init__()
        self.depthwise = DepthSepConvBnAct(in_channels=in_channels, expand=expand, kernel_size=kernel_size, stride=stride, act=act)
        self.pointwise = Conv2dBnAct(in_channels=in_channels * (expand//2) * (expand//2), out_channels=out_channels, kernel_size=1, stride=1, dilation=1,
                                groups=1, padding_mode='zeros', act=act)
        
    def forward(self, input):
        output = self.depthwise(input)
        output = self.pointwise(output)
        return output


class _MicroNet_M2(nn.Module):
    def __init__(self, in_channels, classes):
        super(_MicroNet_M2, self).__init__()
        self.micronetStem = MicroNetStem(in_channels, out_channels=8)

        # configs = block_type, in_channels, kernel_size, out_channels, expand, stride  
        layer1 = [['a', 16, 3, 32, 12, 2]]


        self.layer1 = Make_Layers(layer1)

    def forward(self, input):
        stem = self.micronetStem(input)
        print('# stem shape : ', stem.shape)
        s1 = self.layer1(stem)
        print('# s1 shape : ', s1.shape)


def MicroNet(in_channels, classes=1000, varient=0):
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
