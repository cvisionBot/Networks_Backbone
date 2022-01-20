from ..layers.convolution import Conv2dBnAct
from ..layers.blocks import FrostBlock
from ..initialize import weight_initialize

import torch
from torch import nn

class FrostStem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FrostStem, self).__init__()
        self.conv = Conv2dBnAct(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2)
        
    def forward(self, input):
        output = self.conv(input)
        return output


class Make_Layer(nn.Module):
    def __init__(self, layers_configs):
        super(Make_Layer, self).__init__()
        self.layers_configs = layers_configs
        self.layer = self.frostBlock(self.layers_configs)

    def forward(self, input):
        return self.layer(input)
    
    def frostBlock(self, cfg):
        layers = []
        for i, k, o, e, r, s in cfg:
            layers.append(FrostBlock(in_channels=i, kernel_size=k, out_channels=o, EF=e, RF=r, stride=s))
        return nn.Sequential(*layers)


class _FrostNet(nn.Module):
    def __init__(self, in_channels, classes):
        super(_FrostNet, self).__init__()
        self.frostStem = FrostStem(in_channels=in_channels, out_channels=32)

        # configs : in_channels kernel_size, out_channels, EF, RF, stride
        layer1 = [[32, 3, 16, 1, 1, 1]]
        layer2 = [[16, 5, 24, 6, 4, 2],[24, 3, 24, 3, 4, 1]]
        layer3 = [[24, 5, 40, 3, 4, 2],[40, 5, 40, 3, 4, 1]]
        layer4 = [[40, 5, 80, 3, 4, 2],[80, 3, 80, 3, 4, 1],
                  [80, 5, 96, 3, 2, 1],[96, 3, 96, 3, 4, 1],
                  [96, 5, 96, 3, 4, 1],[96, 5, 96, 3, 4, 1]]
        layer5 = [[96, 5, 192, 6, 2, 2],[192, 5, 192, 3, 2, 1],
                  [192, 5, 192, 3, 2, 1],[192, 5, 192, 3, 2, 1], 
                  [192, 5, 320, 6, 2, 1]]
        
        self.layer1 = Make_Layer(layer1)
        self.layer2 = Make_Layer(layer2)
        self.layer3 = Make_Layer(layer3)
        self.layer4 = Make_Layer(layer4)
        self.layer5 = Make_Layer(layer5)
        self.classification = nn.Sequential(
            Conv2dBnAct(in_channels=320, out_channels=1280, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1280, classes, 1)
        )
        
    def forward(self, input):
        stem= self.frostStem(input)
        s1 = self.layer1(stem)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        s5 = self.layer5(s4)
        pred = self.classification(s5)
        b, c, _, _ = pred.size()
        pred = pred.view(b, c)
        return {'pred':pred}


def FrostNet(in_channels, classes=1000):
    model = _FrostNet(in_channels=in_channels, classes=classes)
    weight_initialize(model)
    return model


if __name__ == '__main__':
    model = FrostNet(in_channels=3, classes=1000)
    model(torch.rand(1, 3, 224, 224))