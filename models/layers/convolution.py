import torch
from torch import nn

def getPadding(kernel_size, mode='same'):
    if mode == 'same':
        return (int((kernel_size - 1) / 2), (int((kernel_size - 1) / 2)))
    else:
        return 0

class Conv2dBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, padding_mode='zeros', act=None):
        super(Conv2dBnAct, self).__init__()
        self.padding = getPadding(kernel_size)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, self.padding, dilation, groups, False, padding_mode)
        self.bn = nn.BatchNorm2d(out_channels)
        if act is None:
            act = nn.ReLU()
        self.act = act

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output

class Conv2dBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, padding_mode='zeros'):
        super(Conv2dBn, self).__init__()
        self.padding = getPadding(kernel_size)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, self.padding, dilation, groups, False, padding_mode)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        output = self.conv(input)
        return self.bn(output)

class DepthwiseConvBnAct(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, dilation=1, padding_mode='zeros', act=None):
        super(DepthwiseConvBnAct, self).__init__()
        self.padding = getPadding(kernel_size)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, self.padding, dilation, in_channels, False, padding_mode)
        self.bn = nn.BatchNorm2d(in_channels)
        if act is None:
            act = nn.ReLU()
        self.act = act

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return self.act(output)

class DepthwiseConvBn(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, dilation=1, padding_mode='zeros'):
        super(DepthwiseConvBn, self).__init__()
        self.padding = getPadding(kernel_size)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, self.padding, dilation, in_channels, False, padding_mode)
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, input):
        output = self.conv(input)
        return self.bn(output)
    
class DenseLayer(nn.Module):
    def __init__(self, in_channels, iter_cnt, growth_rate):
        super(DenseLayer, self).__init__()
        self.iter = iter_cnt
        self.bn_list = nn.ModuleList([])
        self.conv_list = nn.ModuleList([])
        self.relu = nn.ReLU()

        self.in_channels = in_channels
        for i in range(self.iter):
            self.bn_list.append(nn.BatchNorm2d(num_features=self.in_channels))
            self.conv_list.append(nn.Conv2d(in_channels=self.in_channels, out_channels=growth_rate, kernel_size=3, stride=1,
                                                padding=1, dilation=1, groups=1, padding_mode='zeros'))
            self.in_channels = self.in_channels + growth_rate

    def forward(self, input):
        outputs = input
        for bn, cn in zip(self.bn_list, self.conv_list):
            output = bn(outputs)
            output = self.relu(output)
            output = cn(output)
            outputs = torch.cat([outputs, output], axis=1)
        return outputs


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.conv = Conv2dBn(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, input):
        output = self.conv(input)
        output = self.avg_pool(output)
        return output
