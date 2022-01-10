from ..layers.convolution import Conv2dBnAct, DepthwiseConvBnAct
from ..layers.convolution import Conv2dBn, DepthwiseConvBn
from ..layers.convolution import Dense_Layer, Transition_Layer
from ..layers.attention import SE_Block

import torch
from torch import nn


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
    
    
class Dense_Block(nn.Module):
    def __init__(self, in_channels, iter_cnt, transition, growth_rate=32):
        super(Dense_Block, self).__init__()
        self.transition = transition
        self.dense_layer = Dense_Layer(in_channels=in_channels, iter_cnt=iter_cnt, growth_rate=growth_rate)
        self.transition_layaer = Transition_Layer(in_channels=self.calc_channels(in_channels, growth_rate, iter_cnt),
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
    
    
class InvertedResidualBlock(nn.Module): # expansion ratio = t
    def __init__(self, in_channels, out_channels, exp, stride):
        super(InvertedResidualBlock, self).__init__()
        self.stride = stride
        if exp == 0:
            exp = 1
        self.exp = exp
        self.conv1 = Conv2dBnAct(in_channels=in_channels,out_channels=in_channels * self.exp, kernel_size=1, stride=1, dilation=1, 
                                    groups=1, padding_mode='zeros', act=nn.ReLU6())
        self.dconv = DepthwiseConvBnAct(in_channels=in_channels * self.exp, kernel_size=3, stride=self.stride,
                                    dilation=1, padding_mode='zeros', act=nn.ReLU6())
        self.conv2 = Conv2dBn(in_channels=in_channels * self.exp, out_channels=out_channels, kernel_size=1)
        self.identity = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)
        
    def forward(self, input):
        output = self.conv1(input)
        output = self.dconv(output)
        output = self.conv2(output)
        if self.stride == 1:
            input = self.identity(input)
            output = input + output        
        return output