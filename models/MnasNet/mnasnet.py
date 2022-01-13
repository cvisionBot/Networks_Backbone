from ..layers.convolution import Conv2dBnAct
from ..layers.blocks import SepConv, MBConv3, MBConv6
from ..initialize import weight_initialize

import torch
from torch import nn

class MnasNetStem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MnasNetStem, self).__init__()
        self.conv = Conv2dBnAct(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2)

    def forward(self, input):
        return self.conv(input)


class Make_Layers(nn.Module):
    def __init__(self, layers_configs):
        super(Make_Layers, self).__init__()
        self.layers_configs = layers_configs
        self.layer = self.MBConv(self.layers_configs)

    def forward(self, input):
        return self.layer(input)

    def MBConv(self, layers_configs):
        layers = []
        for b, i, k, o, s, in layers_configs:
            if b == 0:
                layers.append(SepConv(in_channels=i, kernel_size=k, out_channels=o, stride=s))
            elif b == 3:
                layers.append(MBConv3(in_channels=i, kernel_size=k, out_channels=o, stride=s))
            elif b == 6:
                layers.append(MBConv6(in_channels=i, kernel_size=k, out_channels=o, stride=s, act=None))
            else:
                layers.append(MBConv6(in_channels=i, kernel_size=k, out_channels=o, stride=s, act=None, SE=True))
        return nn.Sequential(*layers)


class _MnasNet_A1(nn.Module):
    def __init__(self, in_channels, classes):
        super(_MnasNet_A1, self).__init__()
        self.mnasnetStem = MnasNetStem(in_channels=in_channels, out_channels=32)
    
        # Block_type, in_channels, kernel_size, out_channels, stride
        layer1 = [[0, 32, 3, 16, 1], [6, 16, 3, 24, 2], [6, 24, 3, 24, 1]]
        layer2 = [[3, 24, 5, 40, 2], [3, 40, 5, 40, 1], [3, 40, 5, 40, 1]]
        layer3 = [[6, 40, 3, 80, 1], [6, 80, 3, 80, 1], [6, 80, 3, 80, 1],
                  [6, 80, 3, 80, 1], [62, 80, 3, 112, 2], [62, 112, 3, 112, 1]]
        layer4 = [[62, 112, 5, 160, 2], [62, 160, 5, 160, 1], [62, 160, 5, 160, 1],
                  [6, 160, 3, 320, 1]]

        self.layer1 = Make_Layers(layer1)
        self.layer2 = Make_Layers(layer2)
        self.layer3 = Make_Layers(layer3)
        self.layer4 = Make_Layers(layer4)
        self.classification = nn.Sequential(
            Conv2dBnAct(in_channels=320, out_channels=1280, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1280, classes, 1)
        )

    def forward(self, input):
        stem = self.mnasnetStem(input)
        print('# stem shape : ', stem.shape)
        s1 = self.layer1(stem)
        print('# s1 shape : ', s1.shape)
        s2 = self.layer2(s1)
        print('# s2 shape : ', s2.shape)
        s3 = self.layer3(s2)
        print('# s3 shape : ', s3.shape)
        s4 = self.layer4(s3)
        print('# s4 shape : ', s4.shape)
        pred = self.classification(s4)
        b, c, _, _ = pred.size()
        pred = pred.view(b, c)
        return {'pred':pred}
    
def MnasNet(in_channels, classes=1000, varient=1):
    if varient == 1:
        model = _MnasNet_A1(in_channels=in_channels, classes=classes)
    weight_initialize(model)
    return model

if __name__ == '__main__':
    model = MnasNet(in_channels=3, classes=1000, varient=1)
    model(torch.rand(1, 3, 224, 224))