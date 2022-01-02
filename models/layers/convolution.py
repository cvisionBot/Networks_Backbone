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