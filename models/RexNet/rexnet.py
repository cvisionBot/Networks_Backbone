from ..layers.convolution import Conv2dBnAct, Conv2dBn
from ..layers.blocks import RexBlock
from ..initialize import weight_initialize

import torch
from torch import nn


class RexStem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RexStem, self).__init__()
        self.conv = Conv2dBnAct(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2,
                        dilation=1, groups=1, padding_mode='zeros', act=nn.ReLU6())
    
    def forward(self, input):
        output = self.conv(input)
        return output


class Make_Layer(nn.Module):
    def __init__(self, layers_configs):
        super(Make_Layer, self).__init__()
        self.layers_configs = layers_configs
        self.layer = self.rexBlock(self.layers_configs)

    def forward(self, input):
        return self.layer(input)

    def rexBlock(self, cfg):
        layers = []
        for i, k, o, e, s, se in cfg:
            layers.append(RexBlock(in_channels=i, kernel_size=k, out_channels=o, exp=e, stride=s, SE=se))
        return nn.Sequential(*layers)


class _ReXNet(nn.Module):
    def __init__(self, in_channels, classes):
        super(_ReXNet, self).__init__()
        self.rexStem = RexStem(in_channels=3, out_channels=32)

        # configs : in_channels, kernel_size, out_channels, exp, strides, SE
        # don't know why expasion ratio 91 in doc so using ratio 6
        layer1 = [[32, 3, 16, 6, 1, False]]
        layer2 = [[16, 3, 52, 6, 2, False]]
        layer3 = [[52, 3, 68, 6, 2, True]]
        layer4 = [[68, 3, 84, 6, 2, True], [84, 3, 100, 6, 1, True]]
        layer5 = [[100, 3, 116, 6, 2, True]]

        self.layer1 = Make_Layer(layer1)
        self.layer2 = Make_Layer(layer2)
        self.layer3 = Make_Layer(layer3)
        self.layer4 = Make_Layer(layer4)
        self.layer5 = Make_Layer(layer5)
        self.classification = nn.Sequential(
            Conv2dBn(in_channels=116, out_channels=1280, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1280, classes, 1)
        )

    def forward(self, input):
        stem = self.rexStem(input)
        s1 = self.layer1(stem)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        s5 = self.layer5(s4)
        pred = self.classification(s5)
        b, c, _, _ = pred.size()
        pred = pred.view(b, c)
        return {'pred':pred}

def ReXNet(in_channels, classes=1000):
    model = _ReXNet(in_channels=in_channels, classes=classes)
    weight_initialize(model)
    return model

if __name__ == '__main__':
    model = ReXNet(in_channels=3, classes=1000)
    model(torch.rand(1, 3, 224, 224))