from ..layers.convolution import Conv2dBn, Conv2dBnAct
from ..layers.blocks import InvertedResidualBlock
from ..initialize import weight_initialize

import torch
from torch import nn

class MobileNetStem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MobileNetStem, self).__init__()
        self. conv = Conv2dBnAct(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2)

    def forward(self, input):
        return self.conv(input)


class _MobileNetv2(nn.Module):
    def __init__(self, in_channels, classes):
        super(_MobileNetv2, self).__init__()
        self.stage_channels = []
        self.mobilenetStem = MobileNetStem(in_channels=in_channels, out_channels=32)

        # confing in_channels, kernel_size, out_channels, expansion_ratio, stride
        layer1 = [ 
            [32, 3, 16, 1, 1],
            [16, 3, 24, 6, 2], [24, 3, 24, 6, 1]
        ]
        layer2 = [
            [24, 3, 32, 6, 2], [32, 3, 32, 6, 1], [32, 3, 32, 6, 1]
        ]
        layer3 = [ 
            [32, 3, 64, 6, 2], [64, 3, 64, 6, 1], [64, 3, 64, 6, 1], [64, 3, 64, 6, 1],
        ]
        layer4 = [
            [64, 3, 96, 6, 2], [96, 3, 96, 6, 1], [96, 3, 96, 6, 1],
            [96, 3, 160, 6, 1], [160, 3, 160, 6, 1], [160, 3, 160, 6, 1]
        ]
        layer5 = [
            [160, 3, 320, 1, 1]
        ]
        self.layer1 = self.make_layers(layer1)
        self.layer2 = self.make_layers(layer2)
        self.layer3 = self.make_layers(layer3)
        self.layer4 = self.make_layers(layer4)
        self.layer5 = self.make_layers(layer5)
        self.classification = nn.Sequential(
            Conv2dBn(in_channels=320, out_channels=1280, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1280, classes, 1)
        )
    def forward(self, input):
        stem = self.mobilenetStem(input)
        s1 = self.layer1(stem)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        s5 = self.layer5(s4)
        pred = self.classification(s5)
        b, c, _, _ = pred.size()
        pred = pred.view(b, c)
        return {'pred':pred}
    
    def make_layers(self, layers_configs):
        layers = []
        for i, k, o, exp, stride in layers_configs:
            layers.append(InvertedResidualBlock(in_channels=i, kernel_size=k, out_channels=o, exp=exp, stride=stride))
        return nn.Sequential(*layers)


def MobileNetv2(in_channels, classes=1000):
    model = _MobileNetv2(in_channels=in_channels, classes=classes)
    weight_initialize(model)
    return model


if __name__ == '__main__':
    model = MobileNetv2(in_channels=3, classes=1000)
    model(torch.rand(1, 3, 224, 224))