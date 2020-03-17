import torch
import torch.nn as nn
from model_head import BengaliHead
from efficientnet_pytorch import EfficientNet

class SEResNextModel(nn.Module):
    def __init__(self, model, n=[168, 11, 7], pre=True, ps=0.5):
        super().__init__()
        
        w = (model.layer0.conv1.weight.mean(1)).unsqueeze(1)
        model.layer0.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.layer0.conv1.weight = nn.Parameter(w)
        
        self.m = list(model.children())
        nc = self.m[-1].in_features

        self.base = nn.Sequential(*self.m[:-2])
        
        self.head1 = BengaliHead(nc, n[0])
        self.head2 = BengaliHead(nc, n[1])
        self.head3 = BengaliHead(nc, n[2])
        
    def forward(self, x):
        x = self.base(x)
        
        x1 = self.head1(x)
        x2 = self.head2(x)
        x3 = self.head3(x)
        
        return x1,x2,x3
        
class DensenetModel(nn.Module):
    def __init__(self, model, n=[168, 11, 7], ps=0.5):
        super().__init__()

        w = (model.features.conv0.weight.mean(1)).unsqueeze(1)
        model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.features.conv0.weight = nn.Parameter(w)

        self.m = list(model.features.children())
        nc = model.classifier.in_features

        self.base = nn.Sequential(*self.m[:-1])

        self.head1 = BengaliHead(nc, n[0])
        self.head2 = BengaliHead(nc, n[1])
        self.head3 = BengaliHead(nc, n[2])

    def forward(self, x):    
        x = self.base(x)

        x1 = self.head1(x)
        x2 = self.head2(x)
        x3 = self.head3(x)

        return x1,x2,x3

class EfficientNetModel(nn.Module):
    def __init__(self, model_name, n=[168, 11, 7], pre=True, ps=0.5):
        super(EfficientNetModel, self).__init__()
        model = EfficientNet.from_pretrained(model_name)
        
        inp_w = model._conv_stem.weight
        model._conv_stem = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=2, bias=False)
        model._conv_stem.weight = nn.Parameter(torch.mean(inp_w, dim=1, keepdim=True))
        
        self.model = model
        
        m = list(self.model.children())
        nc = m[-2].in_features
        
        self.head1 = BengaliHead(nc,n[0])
        self.head2 = BengaliHead(nc,n[1])
        self.head3 = BengaliHead(nc,n[2])

    def forward(self,x):
        x = self.model.extract_features(x)
        x1 = self.head1(x)
        x2 = self.head2(x)
        x3 = self.head3(x)
        
        return x1,x2,x3
        