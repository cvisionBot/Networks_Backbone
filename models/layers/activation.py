import torch
from torch import nn

class HardSwish(nn.Module):
    def __init__(self):
        super(HardSwish, self).__init__()
        self.relu6 = nn.ReLU6()

    def forward(self, x):
        return x * self.relu6(x+3) / 6

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)

class HardSigmoid(nn.Module):
    def __init__(self, inplace=True, h_max = 1):
        super(HardSigmoid, self).__init__()
        self.relu6 = nn.ReLU6(inplace=True)
        self.h_max = h_max / 6

    def forward(self, x):
        return self.relu6(x + 3) * self.h_max

class DynamicSiftMax(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DynamicSiftMax, self).__init__()
        self.out_channels = out_channels
        self.act_max = 2 # 1.0 * 2
        self.exp = 4
        self.group = 1
        self.squeeze = 4
        self.init_a = [0.0, 0.0]
        self.init_b = [0.0, 0.0]
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, self.squeeze),
            nn.ReLU(inplace=True),
            nn.Linear(self.squeeze, out_channels * self.exp),
            HardSigmoid()
        )
        self.gc = in_channels // self.group
        index=torch.Tensor(range(in_channels)).view(1,in_channels,1,1)
        index=index.view(1,self.group,self.gc,1,1)
        indexgs = torch.split(index, [1, self.group-1], dim=1)
        indexgs = torch.cat((indexgs[1], indexgs[0]), dim=1)
        indexs = torch.split(indexgs, [1, self.gc-1], dim=2)
        indexs = torch.cat((indexs[1], indexs[0]), dim=2)
        self.index = indexs.view(in_channels).type(torch.LongTensor)

    def forward(self, x):
        x_in = x
        x_out = x

        y = self.avg_pool(x_in).view(x_in.size()[0], x_in.size()[1])
        y = self.fc(y).view(x_in.size()[0], self.out_channels * self.exp, 1, 1)
        y = (y - 0.5) * self.act_max

        x2 = x_out[:, self.index, :, :]
        if self.exp == 4:
            a1, b1, a2, b2 = torch.split(y, self.out_channels, dim=1)

            a1 = a1 + self.init_a[0]
            a2 = a2 + self.init_a[1]

            b1 = b1 + self.init_b[0]
            b2 = b2 + self.init_b[1]

            z1 = x_out * a1 + x2 * b1
            z2 = x_out * a2 + x2 * b2

            out = torch.max(z1, z2)
        return out


