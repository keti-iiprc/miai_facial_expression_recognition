from torch import nn
from torch.nn import functional as F
import torch
import torch.nn.init as init
from torchvision import models


class Net(nn.Module):
    def __init__(self, pretrained=True):
        super(Net, self).__init__()
        resnet = models.resnet18(pretrained)
        if pretrained:
            checkpoint = torch.load('./checkpoints/resnet18_msceleb.pth')
            resnet.load_state_dict(checkpoint['state_dict'],strict=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        for i in range(4):
            setattr(self,"layer%d" %i, AttentionBlock())
        self.sig = nn.Sigmoid()
        self.fc = nn.Sequential(nn.Linear(512, 8),
                                  nn.BatchNorm1d(8))
        self.fc_aro = nn.Sequential(nn.Linear(512, 1), 
                                      nn.BatchNorm1d(1))
        self.fc_val = nn.Sequential(nn.Linear(512, 1), 
                                      nn.BatchNorm1d(1))
            
    def get_feature(self, x):
        x = self.features(x)
        return x

    def forward(self, x):
        x = self.get_feature(x)
        layers = []
        for i in range(4):
            layers.append(getattr(self,"layer%d" %i)(x))
        layers = torch.stack(layers).permute([1,0,2])
        if layers.size(1)>1:
            layers = F.log_softmax(layers,dim=1)
        out = self.fc(layers.sum(dim=1))
        aro_out = self.fc_aro(layers.sum(dim=1))
        val_out = self.fc_val(layers.sum(dim=1))
        
        return out, x, layers, aro_out, val_out


class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = SA()
        self.ca = CA()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    def forward(self, x):
        sa = self.sa(x)
        ca = self.ca(sa)

        return ca


class SA(nn.Module):
    def __init__(self):
        super().__init__()
        self.SA_A = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256))
        self.SA_B = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3,padding=1),
            nn.BatchNorm2d(512))
        self.SA_C = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(1,3),padding=(0,1)),
            nn.BatchNorm2d(512))
        self.SA_D = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3,1),padding=(1,0)),
            nn.BatchNorm2d(512))
        self.act = nn.LeakyReLU()

    def forward(self, x):
        xx = self.SA_A(x)
        y = self.act(self.SA_B(xx) + self.SA_C(xx) + self.SA_D(xx))
        out = x * (y.sum(dim=1,keepdim=True))
        return out 


class CA(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.Sequential(
            nn.Linear(512, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(inplace=True),
            nn.Linear(32, 512),
            nn.Sigmoid()    
        )

    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.size(0),-1)
        y = self.attention(x)
        out = x * y        
        return out

