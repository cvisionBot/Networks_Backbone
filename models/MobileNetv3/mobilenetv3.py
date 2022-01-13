from ..layers.convolution import Conv2dBnAct, DepthwiseConvBnAct
from ..layers.blocks import InvertedResidualBlock
from ..layers.attention import SE_Block
from ..layers.activation import HardSwish
from ..initialize import weight_initialize

import torch
from torch import nn

class MobileNetStem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MobileNetStem, self).__init__()
        self.conv = Conv2dBnAct(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, dilation=1,
                                groups=1, padding_mode='zeros', act=HardSwish())
    
    def forward(self, input):
        return self.conv(input)


class Make_Layers(nn.Module):
    def __init__(self, layers_configs):
        super(Make_Layers, self).__init__()
        self.layers_configs = layers_configs
        self.layer = self.Mobilenetv3_layer(self.layers_configs)

    def forward(self, input):
        return self.layer(input)

    def Mobilenetv3_layer(self, layers_configs):
        layers = []
        for i, k , o, e, s, se, nl in layers_configs:
            layers.append(InvertedResidualBlock(in_channels=i, kernel_size=k, out_channels=o,
                                                exp=e, stride=s, SE=se, NL=nl))
        return nn.Sequential(*layers)

class _MobileNetv3_Large(nn.Module):
    def __init__(self, in_channels, classes):
        super(_MobileNetv3_Large, self).__init__()
        self.stage_channels = []
        self.mobilenetStem = MobileNetStem(in_channels=in_channels, out_channels=16)

        # config in_channels, kernel_size, out_channels, exp, stride, SE, NL
        layer1 = [ # 112 x 112 resolution
            [16, 3, 16, 16, 1, False, 'RE'], [16, 3, 24, 64, 2, False, 'RE']
        ]
        layer2 = [ # 56 x 56 resolution
            [24, 3, 24, 72, 1, False, 'RE'], [24, 5, 40, 72, 2, True, 'RE']
        ]
        layer3 = [ # 28 x 28 resolution
            [40, 5, 40, 120, 1, True, 'RE'], [40, 5, 40, 120, 1, True, 'RE'],
            [40, 5, 80, 240, 2, False, 'HS']
        ]
        layer4 = [ # 14 x 14 resolution
            [80, 3, 80, 200, 1, False, 'HS'], [80, 3, 80, 184, 1, False, 'HS'],
            [80, 3, 80, 184, 1, False, 'HS'], [80, 3, 112, 480, 1, True, 'HS'],
            [112, 3, 112, 672, 1, True, 'HS'], [112, 5, 160, 672, 2, True, 'HS']
        ]
        layer5 = [ # 7 x 7 resolution
            [160, 5, 160, 960, 1, True, 'HS'], [160, 5, 160, 960, 1, True, 'HS']
        ]
        self.layer1 = Make_Layers(layer1)
        self.layer2 = Make_Layers(layer2)
        self.layer3 = Make_Layers(layer3)
        self.layer4 = Make_Layers(layer4)
        self.layer5 = Make_Layers(layer5)
        self.classification = nn.Sequential(
            Conv2dBnAct(in_channels=160, out_channels=1280, kernel_size=1),
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

class _MobileNetv3_Small(nn.Module):
    def __init__(self, in_channels, classes):
        super(_MobileNetv3_Small, self).__init__()
        self.stage_channels = []
        self.mobilenetStem = MobileNetStem(in_channels=in_channels, out_channels=16)
     # config in_channels, kernel_size, out_channels, exp, SE, NL, stride
        layer1 = [ # 112 x 112 resolution
            [16, 3, 16, 16, 2, True, 'RE']
        ]
        layer2 = [ # 56 x 56 resolution
            [16, 3, 24, 72, 2, False, 'RE']
        ]
        layer3 = [ # 28 x 28 resolution
            [24, 3, 24, 88, 1, False, 'RE'], [24, 5, 40, 96, 2, True, 'HS']
        ]
        layer4 = [ # 14 x 14 resolution
            [40, 5, 40, 240, 1, True, 'HS'], [40, 5, 40, 240, 1, True, 'HS'],
            [40, 5, 48, 120, 1, True, 'HS'], [48, 5, 48, 144, 1, True, 'HS'],
            [48, 5, 96, 288, 2, True, 'HS']
        ]
        layer5 = [ # 7 x 7 resolution
            [96, 5, 96, 576, 1, True, 'HS'], [96, 5, 96, 576, 1, True, 'HS']
        ]
        self.layer1 = Make_Layers(layer1)
        self.layer2 = Make_Layers(layer2)
        self.layer3 = Make_Layers(layer3)
        self.layer4 = Make_Layers(layer4)
        self.layer5 = Make_Layers(layer5)
        self.classification = nn.Sequential(
            Conv2dBnAct(in_channels=96, out_channels=1280, kernel_size=1),
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

def MobileNetv3(in_channels, classes=1000, varient='small'):
    if varient == 'small':
        model = _MobileNetv3_Small(in_channels=in_channels, classes=classes)
    else:
        model = _MobileNetv3_Large(in_channels=in_channels, classes=classes)
    weight_initialize(model)
    return model

if __name__ == '__main__':
    model = MobileNetv3(in_channels=3, classes=1000, varient='small')
    model(torch.rand(1, 3, 224, 224))