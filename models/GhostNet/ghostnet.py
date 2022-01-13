from ..layers.convolution import Conv2dBnAct
from ..layers.blocks import G_bneck
from ..initialize import weight_initialize

import math
import torch
from torch import nn

class GhostNetStem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GhostNetStem, self).__init__()
        self.conv = Conv2dBnAct(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2)

    def forward(self, input):
        return self.conv(input)


class _GhostNet(nn.Module):
    def __init__(self, in_channels, classes):
        super(_GhostNet, self).__init__()
        self.ghostnetStem = GhostNetStem(in_channels, out_channels=16)

        # config = in_channels, kernel_size, out_channels, exp, stride, SE
        layer1 = [ 
            [16, 3, 16, 16, 1, False], [16, 3, 24, 48, 2, False]
        ]
        layer2 = [
            [24, 3, 24, 72, 1, False], [24, 3, 40, 72, 2, True]
        ]
        layer3 = [ 
            [40, 3, 40, 120, 1, True], [40, 3, 80, 240, 2, False]
        ]
        layer4 = [ 
            [80, 3, 80, 200, 1, False], [80, 3, 80, 184, 1, False],
            [80, 3, 80, 184, 1, False], [80, 3, 112, 480, 1, True],
            [112, 3, 112, 672, 1, True], [112, 3, 160, 672, 2, True]
        ]
        layer5 = [ 
            [160, 3, 160, 960, 1, False], [160, 3, 160, 960, 1, True],
            [160, 3, 160, 960, 1, False], [160, 3, 160, 960, 1, True]
        ]

        self.layer1 = self.make_layer(layer1)
        self.layer2 = self.make_layer(layer2)
        self.layer3 = self.make_layer(layer3)
        self.layer4 = self.make_layer(layer4)
        self.layer5 = self.make_layer(layer5)
        self.classification = nn.Sequential(
            Conv2dBnAct(in_channels=160, out_channels=1280, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1280, classes, 1)
        )

    def forward(self, input):
        stem = self.ghostnetStem(input)
        s1 = self.layer1(stem) 
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        s5 = self.layer5(s4)
        pred = self.classification(s5)
        b, c, _, _ = pred.size()
        pred = pred.view(b, c)
        stages = [s1, s2, s3, s4]
        return {'pred':pred}

    def make_layer(self, layers_configs):
        layers = []
        for i, k, o, e, s, se in layers_configs:
            layers.append(G_bneck(
                in_channels=i, kernel_size=k, out_channels=o, exp=e, stride=s, SE=se
            ))
        return nn.Sequential(*layers)

def GhostNet(in_channels, classes=1000):
    model = _GhostNet(in_channels=in_channels, classes=classes)
    weight_initialize(model)
    return model


if __name__ == '__main__':
    model = GhostNet(in_channels=3, classes=1000)
    model(torch.rand(1, 3, 224, 224))