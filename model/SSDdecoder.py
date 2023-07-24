import torch.nn.init as init
import torch.nn as nn
import torch
from model.Mobilenet import InvertedResidual, MobileNetV2


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        # x /= norm
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x

        return out


def add_extras(in_channels):
    layers = []

    layers += [InvertedResidual(in_channels, 512, stride=2, expand_ratio=0.2)]
    layers += [InvertedResidual(512, 256, stride=2, expand_ratio=0.25)]
    layers += [InvertedResidual(256, 256, stride=2, expand_ratio=0.5)]
    layers += [InvertedResidual(256, 64, stride=2, expand_ratio=0.25)]

    return nn.ModuleList(layers)


class SSD(nn.Module):
    def __init__(self, in_channels=1280, num_classes=10):

        super(SSD, self).__init__()

        self.extras = add_extras(in_channels)
        self.L2Norm = L2Norm(96, 20)

        backbone = MobileNetV2()
        mbox = [6, 6, 6, 6, 6, 6]

        loc_layers = []
        conf_layers = []
        backbone_source = [13, -1]
        for k, v in enumerate(backbone_source):
            loc_layers += [nn.Conv2d(backbone.features[v].out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
            conf_layers += [
                nn.Conv2d(backbone.features[v].out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
        for k, v in enumerate(self.extras, 2):
            loc_layers += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]

        self.loc = nn.ModuleList(loc_layers)
        self.conf = nn.ModuleList(conf_layers)
