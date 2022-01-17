from ..layers.convolution import Conv2dBnAct, Conv2dBn
from ..layers.blocks import MicroBlockA, MicroBlockB, MicroBlockC
from ..layers.activation import DynamicSiftMax
from ..initialize import weight_initialize

import torch
from torch import nn

# Spatial Separable Convolution
class MicroNetStem(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2):
        super(MicroNetStem, self).__init__()
        self.conv = Conv2dBnAct(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2)
    
    def forward(self, input):
        output = self.conv(input)
        return output


class Make_Layers(nn.Module):
    def __init__(self, layers_configs):
        super(Make_Layers, self).__init__()
        self.layers_configs = layers_configs
        self.layer = self.microBlock(self.layers_configs)

    def forward(self, input):
        return self.layer(input)

    def microBlock(self, layers_configs):
        layers = []
        for b, i, k, o, r, g, s in layers_configs:
            if b == 'a':
                layers.append(MicroBlockA(in_channels=i, kernel_size=k, out_channels=o, ratio=r, group=g, stride=s, act=None))
            elif b == 'b':
                layers.append(MicroBlockB(in_channels=i, kernel_size=k, out_channels=o, ratio=r, group=g, stride=s, act=None))
            else:
                layers.append(MicroBlockC(in_channels=i, kernel_size=k, out_channels=o, ratio=r, group=g, stride=s, act=None))
        return nn.Sequential(*layers)


class _MicroNet_M1(nn.Module):
    def __init__(self, in_channels, classes):
        super(_MicroNet_M1, self).__init__()
        self.micronetStem = MicroNetStem(in_channels, out_channels=6)

        # configs = block_type, in_channels, kernel_size, out_channels, Reduction_ratio, Group, stride  
        layer1 = [['a', 6, 3, 24, 3, (2, 0), 2]]
        layer2 = [['a', 8, 3, 32, 2, (4, 0), 2]]
        layer3 = [['b', 16, 5, 96, 6, (4, 4), 2], ['c', 96, 5, 192, 6, (4, 8), 1]]
        layer4 = [['c', 192, 5, 384, 6, (8, 8), 2], ['c', 384, 5, 576, 6, (8, 12), 1]]
        


        self.layer1 = Make_Layers(layer1)
        self.layer2 = Make_Layers(layer2)
        self.layer3 = Make_Layers(layer3)
        self.layer4 = Make_Layers(layer4)
        self.classification = nn.Sequential(
            Conv2dBnAct(in_channels=576, out_channels=1280, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1280, classes, 1)
        )
        
    def forward(self, input):
        stem = self.micronetStem(input)
        s1 = self.layer1(stem)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        pred = self.classification(s4)
        b, c, _, _ = pred.size()
        pred = pred.view(b, c)
        return {'pred':pred}

def MicroNet(in_channels, classes=1000, varient=0):
    model = _MicroNet_M1(in_channels, classes=classes)
    weight_initialize(model)
    return model


if __name__ == '__main__':
    model = MicroNet(in_channels=3, classes=1000, varient=0)
    model(torch.rand(1, 3, 224, 224))