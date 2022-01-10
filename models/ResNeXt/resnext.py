from ..layers.convolution import Conv2dBnAct, Conv2dBn
from ..layers.blocks import ResNeXt_Block
from ..initialize import weight_initialize

import torch
from torch import nn

class ResNeXtStem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNeXtStem, self).__init__()
        self.conv = Conv2dBnAct(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=2)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, input):
        output = self.conv(input)
        output = self.max_pool(output)
        return output


class _ResNeXt50(nn.Module):
    def __init__(self, in_channels, classes):
        super(_ResNeXt50, self).__init__()
        self.resnextStem = ResNeXtStem(in_channels=in_channels, out_channels=64)
        # config in_channels, kernel_size, out_channels, stride, Cardinarity, iter_cnt
        conv2_x = [128, 3, 256, 1, 3]
        conv3_x = [256, 3, 512, 2, 4]
        conv4_x = [512, 3, 1024, 2, 6]
        conv5_x = [1024, 3, 2048, 2, 3]
   
        self.layer1 = self.make_layer(conv2_x, 64)
        self.layer2 = self.make_layer(conv3_x, 256)
        self.layer3 = self.make_layer(conv4_x, 512)
        self.layer4 = self.make_layer(conv5_x, 1024)
        self.classification = nn.Sequential(
            Conv2dBnAct(in_channels=2048, out_channels=1280, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1280, classes, 1)
        )

    def forward(self, input):
        stem= self.resnextStem(input)
        s1 = self.layer1(stem)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        pred = self.classification(s4)
        b, c, _, _ = pred.size()
        pred = pred.view(b, c)
        return {'pred':pred}

    def make_layer(self, cfg, pre_layer_ch):
        layers = []
        self.pre_ch = pre_layer_ch
        input_ch = cfg[0]
        for i in range(cfg[-1]):
            if i == 0:
                layer=ResNeXt_Block(in_channels=self.pre_ch, kernel_size=cfg[1], out_channels=cfg[2], stride=cfg[3])
            else:
                layer = ResNeXt_Block(in_channels=input_ch, kernel_size=cfg[1], out_channels=cfg[2])
            layers.append(layer)
            input_ch = layer.get_channel()
        return nn.Sequential(*layers)

    
def ResNeXt(in_channels, classes=1000):
    model = _ResNeXt50(in_channels=in_channels, classes=classes)
    weight_initialize(model)
    return model

if __name__ == '__main__':
    model = ResNeXt(in_channels=3, classes=1000)
    model(torch.rand(1, 3, 224, 224))