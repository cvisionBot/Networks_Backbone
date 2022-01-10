from torch import nn

class SE_Block(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, act=None):
        super(SE_Block, self).__init__()
        '''
        fully connected Layer - > replace 1 x 1 Convolution Channels
        '''
        self.g_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels//reduction_ratio, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()
        if act is None:
            act = nn.ReLU()
        self.act = act

    def forward(self, input):
        b, c, _, _ = input.size()
        output = self.g_pool(input)
        output = output.view(b, -1)
        output = self.fc1(output)
        if self.act:
            output = self.act(output)
        else:
            output = self.relu(output)
        output = self.fc2(output)
        output = self.sigmoid(output)
        output = output.view(b, c, 1, 1)
        output = input * output
        return output