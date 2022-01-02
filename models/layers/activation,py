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