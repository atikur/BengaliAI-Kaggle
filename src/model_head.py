import torch
import torch.nn as nn
from utils import *

class BengaliHead(nn.Module):
    def __init__(self, nc, n, ps=0.5):
        super().__init__()
        
        self.conv = nn.Conv2d(nc, 512, kernel_size=1, stride=1)
        self.mish = Mish()
        self.bn1 = nn.BatchNorm2d(512)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.flatten = nn.Flatten()
        self.bn2 = nn.BatchNorm1d(512)
        self.drop = nn.Dropout(ps)
        self.lin = nn.Linear(512, n)
        
        self._init_weight()
        
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.mish(x)
        x = self.bn1(x)
        x = 0.5 * (self.avg_pool(x) + self.max_pool(x))
        x = self.mish(x)
        x = self.flatten(x)
        x = self.bn2(x)
        x = self.drop(x)
        x = self.lin(x)
        return x