from ..layers.convolution import Conv2dBnAct, Conv2dBn
from ..layers.blocks import Dense_Block
from ..initialize import weight_initialize

import torch
from torch import nn


class DenseStem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseStem, self).__init__()
        self.conv = Conv2dBnAct(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=2)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, input):
        output = self.conv(input)
        output = self.max_pool(output)
        return output


class Make_Blocks(nn.Module):
    def __init__(self, blocks_configs, transition):
        super(Make_Blocks, self).__init__()
        self.transition = transition
        self.blocks_configs = blocks_configs
        self.block = self.make_dense_block(self.blocks_configs, self.transition)

    def forward(self, input):
        return self.block(input)

    def make_dense_block(self, blocks_configs, transition):
        layers = []
        for i, it in blocks_configs:
            layers.append(Dense_Block(in_channels=i, iter_cnt=it, transition=transition))
        return nn.Sequential(*layers)


class _DenseNet121(nn.Module):
    def __init__(self, in_channels, classes):
        super(_DenseNet121, self).__init__()
        self.denseStem = DenseStem(in_channels=in_channels, out_channels=64)
        
        # config input_channels, iter_cnt
        dense_block1 = [[64, 6]]
        dense_block2 = [[256, 12]]
        dense_block3 = [[640, 24]]
        dense_block4 = [[1408, 16]]

        self.layer1 = Make_Blocks(dense_block1, True)
        self.layer2 = Make_Blocks(dense_block2, True)
        self.layer3 = Make_Blocks(dense_block3, True)
        self.layer4 = Make_Blocks(dense_block4, False)

        self.classification = nn.Sequential(
            Conv2dBnAct(in_channels=1920, out_channels=1280, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1280, classes, 1)
        )

    def forward(self, input):
        stem_out = self.denseStem(input)
        s1 = self.layer1(stem_out)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        pred = self.classification(s4)
        b, c, _, _ = pred.size()
        pred = pred.view(b, c)
        return {'pred':pred}
    


class _DenseNet169(nn.Module):
    def __init__(self, in_channels, classes):
        super(_DenseNet169, self).__init__()
        self.denseStem = DenseStem(in_channels=in_channels, out_channels=64)
        
        # config input_channels, iter_cnt
        dense_block1 = [[64, 6]]
        dense_block2 = [[256, 12]]
        dense_block3 = [[640, 32]]
        dense_block4 = [[1664, 32]]

        self.layer1 = Make_Blocks(dense_block1, True)
        self.layer2 = Make_Blocks(dense_block2, True)
        self.layer3 = Make_Blocks(dense_block3, True)
        self.layer4 = Make_Blocks(dense_block4, False)

        self.classification = nn.Sequential(
            Conv2dBnAct(in_channels=2688, out_channels=1280, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1280, classes, 1)
        )

    def forward(self, input):
        stem = self.denseStem(input)
        s1 = self.layer1(stem)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        pred = self.classification(s4)
        b, c, _, _ = pred.size()
        pred = pred.view(b, c)
        return {'pred':pred}
    
    
    
class _DenseNet201(nn.Module):
    def __init__(self, in_channels, classes):
        super(_DenseNet201, self).__init__()
        self.denseStem = DenseStem(in_channels=in_channels, out_channels=64)
        
        # config input_channels, iter_cnt
        dense_block1 = [[64, 6]]
        dense_block2 = [[256, 12]]
        dense_block3 = [[640, 48]]
        dense_block4 = [[2176, 32]]

        self.layer1 = Make_Blocks(dense_block1, True)
        self.layer2 = Make_Blocks(dense_block2, True)
        self.layer3 = Make_Blocks(dense_block3, True)
        self.layer4 = Make_Blocks(dense_block4, False)

        self.classification = nn.Sequential(
            Conv2dBnAct(in_channels=3200, out_channels=1280, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1280, classes, 1)
        )

    def forward(self, input):
        stem = self.denseStem(input)
        s1 = self.layer1(stem)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        pred = self.classification(s4)
        b, c, _, _ = pred.size()
        pred = pred.view(b, c)
        return {'pred':pred}
    
    

class _DenseNet264(nn.Module):
    def __init__(self, in_channels, classes):
        super(_DenseNet264, self).__init__()
        self.denseStem = DenseStem(in_channels=in_channels, out_channels=64)
        
        # config input_channels, iter_cnt
        dense_block1 = [[64, 6]]
        dense_block2 = [[256, 12]]
        dense_block3 = [[640, 64]]
        dense_block4 = [[2688, 48]]

        self.layer1 = Make_Blocks(dense_block1, True)
        self.layer2 = Make_Blocks(dense_block2, True)
        self.layer3 = Make_Blocks(dense_block3, True)
        self.layer4 = Make_Blocks(dense_block4, False)

        self.classification = nn.Sequential(
            Conv2dBnAct(in_channels=4224, out_channels=1280, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1280, classes, 1)
        )

    def forward(self, input):
        stem = self.denseStem(input)
        s1 = self.layer1(stem)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        pred = self.classification(s4)
        b, c, _, _ = pred.size()
        pred = pred.view(b, c)
        return {'pred':pred}
    

def DenseNet(in_channels, classes=1000, varient=121):
    if varient == 121:
        model = _DenseNet121(in_channels=in_channels, classes=classes)
    elif varient == 169:
        model = _DenseNet169(in_channels=in_channels, classes=classes)
    elif varient == 201:
        model = _DenseNet201(in_channels=in_channels, classes=classes)
    elif varient == 264:
        model = _DenseNet264(in_channels=in_channels, classes=classes)
    else:
        raise Exception('No Such models DenseNet_{}'.format(varient))
    weight_initialize(model)
    return model

if __name__ == '__main__':
    model = DenseNet(in_channels=3, classes=1000, varient=264)
    model(torch.rand(1, 3, 224, 224))