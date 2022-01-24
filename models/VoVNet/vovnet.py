from ..layers.convolution import Conv2dBnAct
from ..layers.blocks import OSAModule
from ..initialize import weight_initialize

import torch
from torch import nn

class VoVStem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VoVStem, self).__init__()
        self.conv = Conv2dBnAct(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2)
    
    def forward(self, input):
        output = self.conv(input)
        return output


class VoVBlock(nn.Module):
    def __init__(self, in_channels, conv_channels, layers_per_block, trans_ch):
        super(VoVBlock, self).__init__()
        self.max_pool = nn.MaxPool2d(2, 2, 0)
        self.osa = OSAModule(in_channels=in_channels, conv_channels=conv_channels, layers_per_block=layers_per_block, trans_ch=trans_ch)

    def forward(self, input):
        output = self.max_pool(input)
        output = self.osa(output)
        return output


class Make_Layer(nn.Module):
    def __init__(self, layers_configs):
        super(Make_Layer, self).__init__()
        self.layers_configs = layers_configs
        self.layer = self.vovBlock(self.layers_configs)

    def forward(self, input):
        return self.layer(input)

    def vovBlock(self, cfg):
        layers = []
        for i, c, n, t in cfg:
            layers.append(VoVBlock(in_channels=i, conv_channels=c, layers_per_block=n, trans_ch=t))
        return nn.Sequential(*layers)


class _VoVNet19(nn.Module):
    def __init__(self, in_channels, classes):
        super(_VoVNet19, self).__init__()
        self.vovStem = VoVStem(in_channels=3, out_channels=64)

        # configs OSA : in_channels, conv, iter_cnt, trans_channels
        layer3 = [[128, 80, 3, 256]]
        layer4 = [[256, 96, 3, 384]]
        layer5 = [[384, 112, 3, 512]]
        
        self.layer1 = Conv2dBnAct(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.layer2 =  nn.Sequential(
                        Conv2dBnAct(in_channels=64, out_channels=128, kernel_size=3, stride=2),
                        OSAModule(in_channels=128, conv_channels=64, layers_per_block=5, trans_ch=128)
                    )
        self.layer3 = Make_Layer(layer3)
        self.layer4 = Make_Layer(layer4)
        self.layer5 = Make_Layer(layer5)
        self.classification = nn.Sequential(
            Conv2dBnAct(in_channels=512, out_channels=1280, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1280, classes, 1)
        )

    def forward(self, input):
        stem= self.vovStem(input)
        s1 = self.layer1(stem)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        s5 = self.layer5(s4)
        pred = self.classification(s5)
        b, c, _, _ = pred.size()
        pred = pred.view(b, c)
        return {'pred':pred}


class _VoVNet27(nn.Module):
    def __init__(self, in_channels, classes):
        super(_VoVNet27, self).__init__()
        self.vovStem = VoVStem(in_channels=3, out_channels=64)

        # configs OSA : in_channels, conv, iter_cnt, trans_channels
        layer3 = [[128, 80, 5, 256]]
        layer4 = [[256, 96, 5, 384]]
        layer5 = [[384, 112, 5, 512]]
        
        self.layer1 = Conv2dBnAct(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.layer2 =  nn.Sequential(
                        Conv2dBnAct(in_channels=64, out_channels=128, kernel_size=3, stride=2),
                        OSAModule(in_channels=128, conv_channels=64, layers_per_block=5, trans_ch=128)
                    )
        self.layer3 = Make_Layer(layer3)
        self.layer4 = Make_Layer(layer4)
        self.layer5 = Make_Layer(layer5)
        self.classification = nn.Sequential(
            Conv2dBnAct(in_channels=512, out_channels=1280, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1280, classes, 1)
        )

    def forward(self, input):
        stem= self.vovStem(input)
        s1 = self.layer1(stem)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        s5 = self.layer5(s4)
        pred = self.classification(s5)
        b, c, _, _ = pred.size()
        pred = pred.view(b, c)
        return {'pred':pred}



def VoVNet(in_channels, classes=1000, varient=27):
    if varient == 19:
        model = _VoVNet19(in_channels=in_channels, classes=classes)
    elif varient == 27:
        model = _VoVNet27(in_channels=in_channels, classes=classes)
    
    weight_initialize(model)
    return model


if __name__ == '__main__':
    model = VoVNet(in_channels=3, classes=1000, varient=27)
    model(torch.rand(1, 3, 224, 224))