from ..layers.convolution import Conv2dBnAct
from ..layers.blocks import DepthwiseSeparable_Block
from ..initialize import weight_initialize

import torch
from torch import nn

class MobileNetStem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MobileNetStem, self).__init__()
        self.conv = Conv2dBnAct(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2)

    def forward(self, input):
        return self.conv(input)

class _MobileNetv1(nn.Module):
    def __init__(self, in_channels, classes):
        super(_MobileNetv1, self).__init__()
        self.stage_channels = []
        self.mobilenetStem = MobileNetStem(in_channels=in_channels, out_channels=32)
        
        # confing in_channels, kernel_size, output_ch, stride
        layer1 = [ 
            [32, 3, 64, 1], [64, 3, 128, 2]
        ]
        layer2 = [ 
            [128, 3, 128, 1], [128, 3, 256, 2]
        ]
        layer3 = [ 
            [256, 3, 256, 1], [256, 3, 512, 2]
        ]
        layer4 = [ 
            [512, 3, 512, 1], [512, 3, 512, 1],
            [512, 3, 512, 1], [512, 3, 512, 1],
            [512, 3, 512, 1], [512, 3, 1024, 2]
        ]
        layer5 = [
            [1024, 3, 1024, 1]
        ]
        self.layer1 = self.make_layers(layer1)
        self.layer2 = self.make_layers(layer2)
        self.layer3 = self.make_layers(layer3)
        self.layer4 = self.make_layers(layer4)
        self.layer5 = self.make_layers(layer5)
        self.stage_channels = self.Get_Stage_Channels([layer1, layer2, layer3, layer4, layer5])
        self.classification = nn.Sequential(
            Conv2dBnAct(in_channels=1024, out_channels=1024, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1024, classes, 1)
        )
    
    def forward(self, input):
        stem_out = self.mobilenetStem(input)      
        s1 = self.layer1(stem_out)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        s5 = self.layer5(s4)
        pred = self.classification(s5)
        b, c, _, _ = pred.size()
        pred = pred.view(b, c)
        stages = [s1, s2, s3, s4, s5]
        return {'stage':stages, 'pred':pred}

    # confing in_channels, kernel_size, output_ch, stride
    def make_layers(self, cfg):
        layers = []
        for i, k, o, s in cfg:
            layer = DepthwiseSeparable_Block(in_channels=i, kernel_size=k, out_channels=o, stride=s)
            layers.append(layer)
        return nn.Sequential(*layers)


def MobileNetv1(in_channels, classes=1000):
    model = _MobileNetv1(in_channels=in_channels, classes=classes)
    weight_initialize(model)
    return model


if __name__ == '__main__':
    model = MobileNetv1(in_channels=3, classes=1000)
    model(torch.rand(1, 3, 224, 224))