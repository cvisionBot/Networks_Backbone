from ..layers.convolution import Conv2dBnAct, DepthwiseConvBnAct
from ..layers.convolution import Conv2dBn, DepthwiseConvBn
from ..layers.convolution import DenseLayer, TransitionLayer
from ..layers.convolution import DepthSepConvBnAct
from ..layers.attention import SE_Block
from ..layers.activation import HardSwish, Swish

import math
import torch
from torch import nn

# ResNet Module
class Residual_Block(nn.Module):
    def __init__(self, in_channels, kernel_size, out_channels, stride=1, SE=False):
        super(Residual_Block, self).__init__()
        self.SE = SE
        self.out_channels = out_channels
        self.res_conv1 = Conv2dBnAct(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1)
        self.res_conv2 = Conv2dBnAct(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=1)
        self.res_conv3 = Conv2dBn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)
        self.identity = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)
        self.se_block = SE_Block(in_channels=out_channels)
        
    def forward(self, input):
        output = self.res_conv1(input)
        output = self.res_conv2(output)
        output = self.res_conv3(output)
        if input.size() != output.size():
            input = self.identity(input)
        output = input + output
        if self.SE:
            output = self.se_block(output)
        return output
    
    def get_channel(self):
        return self.out_channels
    
# ResNet Lite Module    
class Residual_LiteBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, out_channels, stride=1):
        super(Residual_LiteBlock, self).__init__()
        self.out_channels = out_channels
        self.res_conv1 = Conv2dBnAct(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=1)
        self.res_conv2 = Conv2dBn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        self.identity = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)
        
    def forward(self, input):
        output = self.res_conv1(input)
        output = self.res_conv2(output)
        if input.size() != output.size():
            input = self.identity(input)
        output = input + output
        return output

    def get_channel(self):
        return self.out_channels

# ResNeXt Module
class ResNeXt_Block(nn.Module):
    def __init__(self, in_channels, kernel_size, out_channels, stride=1, cardinarity=32, SE=False):
        super(ResNeXt_Block, self).__init__()
        self.SE = SE
        self.cardinarity = cardinarity
        self.out_channels = out_channels
        self.res_conv1 = Conv2dBnAct(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1)
        self.group_conv = Conv2dBnAct(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=1, dilation=1, groups=self.cardinarity)
        self.res_conv2 = Conv2dBnAct(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)
        self.identity = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)
        self.se_block = SE_Block(in_channels=out_channels)

    def forward(self, input):
        output = self.res_conv1(input)
        output = self.group_conv(output)
        output = self.res_conv2(output)
        if input.size() != output.size():
            input = self.identity(input)
        output = input + output
        if self.SE:
            output = self.se_block(output)
        return output
    
    def get_channel(self):
        return self.out_channels
    
# MobileNetv1 Module
class DepthwiseSeparable_Block(nn.Module):
    def __init__(self, in_channels, kernel_size, out_channels, stride):
        super(DepthwiseSeparable_Block, self).__init__()
        self.out_channels = out_channels
        self.depthwise = DepthwiseConvBnAct(in_channels=in_channels, kernel_size=kernel_size, stride=stride)
        self.pointwise = Conv2dBnAct(in_channels = in_channels, out_channels=out_channels, kernel_size=1, stride=1)
    
    def forward(self, input):
        output = self.depthwise(input)
        output = self.pointwise(output)
        return output
    
# DenseNet Module
class Dense_Block(nn.Module):
    def __init__(self, in_channels, iter_cnt, transition, growth_rate=32):
        super(Dense_Block, self).__init__()
        self.transition = transition
        self.dense_layer = DenseLayer(in_channels=in_channels, iter_cnt=iter_cnt, growth_rate=growth_rate)
        self.transition_layaer = TransitionLayer(in_channels=self.calc_channels(in_channels, growth_rate, iter_cnt),
                                                    out_channels=self.calc_channels(in_channels, growth_rate, iter_cnt))

    def forward(self, input):
        output = self.dense_layer(input)
        if self.transition:
            output = self.transition_layaer(output)
        # print('output ch : ', output.size()[1])
        return output
    
    def calc_channels(self, in_channels, growth_rate, layer_len):
        cat_channels = in_channels + (growth_rate * layer_len)
        return cat_channels
    
# MobileNetv2 ~ v3 Module
class InvertedResidualBlock(nn.Module): # expansion ratio = t
    def __init__(self, in_channels, kernel_size, out_channels, exp, stride, SE=False, NL='RE'):
        super(InvertedResidualBlock, self).__init__()
        if exp == 0:
            exp = 1
        self.exp = exp
        self.stride = stride
        self.se_block = SE
        if NL == 'RE':
            self.act = nn.ReLU6()
        else:
            self.act = HardSwish()
        self.conv1 = Conv2dBnAct(in_channels=in_channels,out_channels=in_channels * self.exp, kernel_size=1, stride=1, dilation=1, 
                                    groups=1, padding_mode='zeros', act=self.act)
        self.dconv = DepthwiseConvBnAct(in_channels=in_channels * self.exp, kernel_size=kernel_size, stride=self.stride,
                                    dilation=1, padding_mode='zeros', act=self.act)
        self.conv2 = Conv2dBn(in_channels=in_channels * self.exp, out_channels=out_channels, kernel_size=1)
        self.se = SE_Block(in_channels=in_channels * self.exp)
        self.identity = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)
        

    def forward(self, input):
        output = self.conv1(input)
        output = self.dconv(output)
        if self.se_block:
            se = self.se(output)
            output = self.conv2(se)
        else:
            output = self.conv2(output)
        if self.stride == 1:
            input = self.identity(input)
            output = input + output        
        return output

# GhostNet Module
class G_bneck(nn.Module):
    def __init__(self, in_channels, kernel_size, out_channels, exp, stride, SE, ratio=2):
        super(G_bneck, self).__init__()
        self.stride=stride
        self.use_se = SE
        self.gmodule1 = G_module(in_channels=in_channels, kernel_size=1, out_channels=exp, stride=1)
        self.gmodule2 = G_module(in_channels=exp, kernel_size=1, out_channels=out_channels, stride=1)
        self.dconv = DepthwiseConvBn(in_channels=exp, kernel_size=kernel_size, stride=self.stride)
        self.se_block = SE_Block(in_channels=exp ,reduction_ratio=16)

    def forward(self, input):
        output = self.gmodule1(input)
        if self.stride == 2:
            output = self.dconv(output)
        if self.use_se:
            output = self.se_block(output)
        output = self.gmodule2(output)
        return output 


class G_module(nn.Module):
    def __init__(self, in_channels, kernel_size, out_channels, stride, ratio=2):
        super(G_module, self).__init__()
        self.init_channels = math.ceil(out_channels//ratio)
        self.conv = Conv2dBnAct(in_channels=in_channels, out_channels=self.init_channels, kernel_size=kernel_size, stride=stride)
        self.dconv = DepthwiseConvBn(in_channels=self.init_channels, kernel_size=3, stride=stride)

    def forward(self, input):
        output = self.conv(input)
        depth_output = self.dconv(output)
        output = torch.cat([output, depth_output], axis=1)
        return output

# RegNet Module
class XBlock(nn.Module):
    def __init__(self, in_channels, block_width, bottleneck_ratio, stride, groups=1):
        super(XBlock, self).__init__()
        self.B_conv1 = Conv2dBnAct(in_channels, int(block_width / bottleneck_ratio), 1)
        self.B_conv2 = Conv2dBnAct(int(block_width / bottleneck_ratio), int(block_width / bottleneck_ratio), 3, stride, 1, groups);
        self.B_conv3 = Conv2dBnAct(int(block_width / bottleneck_ratio), block_width, 1)
        self.se_block = SE_Block(int(block_width / bottleneck_ratio))
        self.identity = Conv2dBnAct(in_channels, block_width, 1, stride)

    def forward(self, input):
        output = self.B_conv1(input)
        output = self.B_conv2(output)
        output = self.B_conv3(output)
        if input.size() != output.size():
            input = self.identity(input)
        return output

class YBlock(nn.Module):
    def __init__(self, in_channels, block_width, bottleneck_ratio, stride, groups=1):
        super(YBlock, self).__init__()
        self.B_conv1 = Conv2dBnAct(in_channels, int(block_width / bottleneck_ratio), 1)
        self.B_conv2 = Conv2dBnAct(int(block_width / bottleneck_ratio), int(block_width / bottleneck_ratio), 3, stride, 1, groups);
        self.B_conv3 = Conv2dBnAct(int(block_width / bottleneck_ratio), block_width, 1)
        self.se_block = SE_Block(int(block_width / bottleneck_ratio))
        self.identity = Conv2dBnAct(in_channels, block_width, 1, stride)

    def forward(self, input):
        output = self.B_conv1(input)
        output = self.B_conv2(output)
        output = self.se_block(output)
        output = self.B_conv3(output)
        if input.size() != output.size():
            input = self.identity(input)
        return output

# MNasNet EfficientNet Module
class MBConv1(nn.Module):
    def __init__(self, in_channels, kernel_size, out_channels, stride):
        super(MBConv1, self).__init__()
        self.conv_act = Conv2dBnAct(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, dilation=1,
                                    groups=1, padding_mode='zeros', act=Swish()) 
        self.dconv = DepthwiseConvBnAct(in_channels=in_channels, kernel_size=kernel_size, stride=1, dilation=1,
                                    padding_mode='zeros', act=Swish())
        self.se = SE_Block(in_channels=in_channels)
        self.conv_bn = Conv2dBn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)
        self.identity=nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)
    
    def forward(self, input):
        output = self.conv_act(input)
        output = self.dconv(output)
        output = self.se(output)
        output = self.conv_bn(output)
        if input.size() != output.size():
            input = self.identity(input)
        output = input + output
        return output


class SepConv(nn.Module):
    def __init__(self, in_channels, kernel_size, out_channels, stride):
        super(SepConv, self).__init__()
        self.dconv = DepthwiseConvBnAct(in_channels=in_channels, kernel_size=kernel_size, stride=stride)
        self.conv = Conv2dBn(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
    
    def forward(self, input):
        output = self.dconv(input)
        output = self.conv(output)
        return output


class MBConv3(nn.Module):
    def __init__(self, in_channels, kernel_size, out_channels, stride):
        super(MBConv3, self).__init__()
        self.conv1 = Conv2dBnAct(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.dconv = DepthwiseConvBnAct(in_channels=in_channels, kernel_size=kernel_size, stride=stride)
        self.se = SE_Block(in_channels=in_channels)
        self.conv2 = Conv2dBn(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.identity = Conv2dBn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)

    def forward(self, input):
        output = self.conv1(input)
        output = self.dconv(output)
        output = self.se(output)
        output = self.conv2(output)
        if input.size() != output.size():
            input = self.identity(input)
        output = output + input
        return output


class MBConv6(nn.Module):
    def __init__(self, in_channels, kernel_size, out_channels, stride, act=None, SE=False):
        super(MBConv6, self).__init__()
        if act is None:
            act == nn.ReLU()
        else:
            act = act
        self.act = act
        self.se=SE
        self.conv1 = Conv2dBnAct(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1,
                                groups=1, padding_mode='zeros', act=self.act)
        self.dconv = DepthwiseConvBnAct(in_channels=in_channels, kernel_size=kernel_size, stride=stride, dilation=1, padding_mode='zeros', act=self.act)
        self.conv2 = Conv2dBn(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.se_block = SE_Block(in_channels=in_channels)
        self.identity = Conv2dBn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)

    def forward(self, input):
        output = self.conv1(input)
        output = self.dconv(output)
        if self.se:
            output = self.se_block(output)
        output = self.conv2(output)
        if input.size() != output.size():
            input = self.identity(input)
        output = output + input
        return output

# MicroNet Module
class MicroBlockA(nn.Module):
    def __init__(self, in_channels, kernel_size, out_channels, ratio, group, stride, act=None):
        super(MicroBlockA, self).__init__()
        self.depthwise = DepthSepConvBnAct(in_channels=in_channels, expand=out_channels, kernel_size=kernel_size, stride=stride, act=act)
        self.pointwise = Conv2dBnAct(in_channels=self.depthwise.get_channels(), out_channels=out_channels//ratio, kernel_size=1, stride=1, dilation=1,            
               groups=group[0], padding_mode='zeros', act=act)

        
    def forward(self, input):
        output = self.depthwise(input)
        output = self.pointwise(output)
        return output


class MicroBlockB(nn.Module):
    def __init__(self, in_channels, kernel_size, out_channels, ratio, group, stride, act=None):
        super(MicroBlockB, self).__init__()
        self.depthwise = DepthSepConvBnAct(in_channels=in_channels, expand=out_channels, kernel_size=kernel_size, stride=stride, act=act)
        self.pointwise1 = Conv2dBnAct(in_channels=self.depthwise.get_channels(), out_channels=out_channels//ratio, kernel_size=1, stride=1, dilation=1,            
                groups=group[0], padding_mode='zeros', act=act)
        self.pointwise2 = Conv2dBnAct(in_channels=out_channels//ratio, out_channels=out_channels, kernel_size=1, stride=1, dilation=1,
                groups=group[1], padding_mode='zeros', act=act)
        
    def forward(self, input):
        output = self.depthwise(input)
        output = self.pointwise1(output)
        output = self.pointwise2(output)
        return output


class MicroBlockC(nn.Module):
    def __init__(self, in_channels, kernel_size, out_channels, ratio, group, stride, act=None):
        super(MicroBlockC, self).__init__()
        self.out_channels = out_channels // ratio
        self.depthwise = DepthwiseConvBnAct(in_channels=in_channels, kernel_size=kernel_size, stride=stride, dilation=1,
                                            padding_mode='zeros', act=nn.ReLU6())
        self.pointwise1 = Conv2dBnAct(in_channels=in_channels, out_channels=out_channels//ratio, kernel_size=1, stride=1, dilation=1,            
                groups=group[0], padding_mode='zeros', act=act)
        self.pointwise2 = Conv2dBnAct(in_channels=out_channels//ratio, out_channels=out_channels, kernel_size=1, stride=1, dilation=1,
                groups=group[1], padding_mode='zeros', act=act)
        self.identity = Conv2dBn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)
        
    def forward(self, input):
        output = self.depthwise(input)
        output = self.pointwise1(output)
        output = self.pointwise2(output)
        if input.size() != output.size():
            input = self.identity(input)
        output = output + input
        return output

# FrostNet Module
class FrostBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, out_channels, EF, RF, stride):
        super(FrostBlock, self).__init__()
        self.squeeze_channels = (in_channels // RF)
        self.expansion_channels = (in_channels + self.squeeze_channels) * EF
        self.squeeze = Conv2dBnAct(in_channels=in_channels, out_channels=self.squeeze_channels, kernel_size=1, stride=1)
        self.conv1 = Conv2dBnAct(in_channels=in_channels + self.squeeze_channels, out_channels= self.expansion_channels, kernel_size=1, stride=1)
        self.depthconv = DepthwiseConvBnAct(in_channels=self.expansion_channels, kernel_size=kernel_size, stride=stride)
        self.conv2 = Conv2dBn(in_channels=self.expansion_channels, out_channels=out_channels, kernel_size=1, stride=1)
        self.identity = Conv2dBn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)

    def forward(self, input):
        output = self.squeeze(input)
        output = torch.cat([output, input], axis=1)
        output = self.conv1(output)
        output = self.depthconv(output)
        output = self.conv2(output)
        if input.size() != output.size():
            input = self.identity(input)
        output = output + input
        return output