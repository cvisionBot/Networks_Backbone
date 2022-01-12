from ..layers.convolution import Conv2dBnAct
from ..layers.blocks import XBlock, YBlock
from ..initialize import weight_initialize

import torch
from torch import nn
import numpy as np

class Make_XStage(nn.Module):
    def __init__(self, stage_configs):
        super(Make_XStage, self).__init__()
        self.stage_configs = stage_configs
        self.stage = self.stage_block(self.stage_configs)
    
    def forward(self, input):
        return self.stage(input)

    def stage_block(self, stage_configs):
        stages = []
        for sd, i, bw, br, g in stage_configs:
            stages.append(RegNetXStage(
                sd, i, bw, br, g
            ))
        return nn.Sequential(*stages)

class Make_YStage(nn.Module):
    def __init__(self, stage_configs):
        super(Make_YStage, self).__init__()
        self.stage_configs = stage_configs
        self.stage = self.stage_block(self.stage_configs)
    
    def forward(self, input):
        return self.stage(input)

    def stage_block(self, stage_configs):
        stages = []
        for sd, i, bw, br, g in stage_configs:
            stages.append(RegNetYStage(
                sd, i, bw, br, g
            ))
        return nn.Sequential(*stages)

class RegNetXStage(nn.Module):
    def __init__(self, block_num, in_channels, block_width, bottleneck_ratio, groups):
        super(RegNetXStage, self).__init__()
        self.stage = nn.ModuleList([])
        for index in range(0, block_num):
            if index == 0:
                self.stage.append(XBlock(in_channels, block_width, bottleneck_ratio, 2, groups))
            else:
                self.stage.append(XBlock(block_width, block_width, bottleneck_ratio, 1, groups))

    def forward(self, input):
        output = input
        for block in self.stage:
            output = block(output)
        return output


class RegNetYStage(nn.Module):
    def __init__(self, block_num, in_channels, block_width, bottleneck_ratio, groups):
        super(RegNetYStage, self).__init__()
        self.stage = nn.ModuleList([])
        for index in range(0, block_num):
            if index == 0:
                self.stage.append(YBlock(in_channels, block_width, bottleneck_ratio, 2, groups))
            else:
                self.stage.append(YBlock(block_width, block_width, bottleneck_ratio, 1, groups))

    def forward(self, input):
        output = input
        for block in self.stage:
            output = block(output)
        return output

class _RegNetX_200MF(nn.Module):
    def __init__(self, in_channels, classes):
        super(_RegNetX_200MF, self).__init__()
        # config stage_depth, in_channels, block_width, bottleneck_ratio, group
        '''
        self.stage_depth = [1, 1, 1, 1]
        self.block_width=[128, 256, 512, 1024]
        '''
        # config stage_depth, in_channels, block_width, bottleneck_ratio, group
        stageX1= [[1, 3, 24, 1, 8]]
        stageX2 =[[1, 24, 56, 1, 8]]
        stageX3 = [[4, 56, 152, 1, 8]]
        stageX4 =[[7, 152, 368, 1, 8]]

        self.stage1 = Make_XStage(stageX1)
        self.stage2 = Make_XStage(stageX2)
        self.stage3 = Make_XStage(stageX3)
        self.stage4 = Make_XStage(stageX4)
        self.classifier = nn.Sequential(
            Conv2dBnAct(368, 1280, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1280, classes, 1)
        )
    
    def forward(self, input):
        s1 = self.stage1(input)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        pred = self.classifier(s4)
        b, c, _, _ = pred.size()
        pred = pred.view(b, c)
        return {'pred': pred}


class _RegNetY_200MF(nn.Module):
    def __init__(self, in_channels, classes):
        super(_RegNetY_200MF, self).__init__()
        # config stage_depth, in_channels, block_width, bottleneck_ratio, group
        stageY1= [[1, 3, 24, 1, 8]]
        stageY2 =[[1, 24, 56, 1, 8]]
        stageY3 = [[4, 56, 152, 1, 8]]
        stageY4 =[[7, 152, 368, 1, 8]]

        self.stage1 = Make_YStage(stageY1)
        self.stage2 = Make_YStage(stageY2)
        self.stage3 = Make_YStage(stageY3)
        self.stage4 = Make_YStage(stageY4)
        self.classifier = nn.Sequential(
            Conv2dBnAct(368, 1280, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1280, classes, 1)
        )
    
    def forward(self, input):
        s1 = self.stage1(input)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        pred = self.classifier(s4)
        b, c, _, _ = pred.size()
        pred = pred.view(b, c)
        return {'pred': pred}

def RegNet(in_channels, classes=1000, varient='X'):
    if varient == 'X':
        model = _RegNetX_200MF(in_channels=in_channels, classes=classes)
    else:
        model = _RegNetY_200MF(in_channels=in_channels, classes=classes)
    weight_initialize(model)
    return model


if __name__=='__main__':
    model = RegNet(in_channels=3, classes=1000, varient='X')
    model(torch.rand(1, 3, 224, 224))