import torch
import torch.nn.functional as F
import torch.nn as nn
from model.Mobilenet import *
from model.SSDdecoder import *
from model.DepthDecoder import Depth_decoder


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# Bottleneck from the official implementation of ResNet
class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out


class MTANMobilenet(nn.Module):
    def __init__(self, num_classes=9):
        super(MTANMobilenet, self).__init__()
        # backbone = mobilenet_v2(pretrained=True)
        backbone = mobilenet_v2(pretrained=True)
        self.tasks = ['detection', 'depth']

        self.logsigma = nn.Parameter(torch.FloatTensor([1.5192, -0.5720]))
        # Apply the attention on the last bottleneck layer

        self.shared_layer0_b = nn.Sequential(
            backbone.features[0], backbone.features[1].conv[0])
        self.shared_layer0_t = nn.Sequential(
            backbone.features[1].conv[1], backbone.features[1].conv[2])
        # n = 2
        self.shared_layer1_b = backbone.features[2]  # 16-24
        self.shared_layer1_t = backbone.features[3]  # 24-24
        # n = 3
        self.shared_layer2_b = backbone.features[4:6]  # 24-32
        self.shared_layer2_t = backbone.features[6]  # 32-32
        # n = 4
        self.shared_layer3_b = backbone.features[7:10]  # 32-64
        self.shared_layer3_t = backbone.features[10]  # 64-64
        # n = 3
        self.shared_layer4_b = backbone.features[11:13]  # 64-96
        self.shared_layer4_t = backbone.features[13]  # 96-96
        # n = 3
        self.shared_layer5_b = backbone.features[14:16]  # 96-160
        self.shared_layer5_t = backbone.features[16]  # 160-160
        # n = 1
        # self.shared_layer6_b = backbone.features[17].conv[0:2]  # 160-960
        # self.shared_layer6_t = backbone.features[17].conv[2:4]   # 960-320
        self.shared_layer6_b = backbone.features[17]  # 160-320
        self.shared_layer6_t = backbone.features[-1]  # 320-1280
        # # self.shared_layer7 = backbone.features[-1]  # 320-1240
        # # Define task sepecific attention module
        # # We do not apply shared attention encoders at the last layer,
        # # so the attended features will be directly fed into the task-specific decoders.
        # self.encoder_att_0 = nn.ModuleList([self.att_layer(32, 32 // 4, 16) for _ in self.tasks])
        # self.encoder_att_1 = nn.ModuleList([self.att_layer(24*2, 24 // 4, 24) for _ in self.tasks])
        # self.encoder_att_2 = nn.ModuleList([self.att_layer(32 * 2, 64 // 4, 32) for _ in self.tasks])
        # self.encoder_att_3 = nn.ModuleList([self.att_layer(64 * 2, 128 // 4, 64) for _ in self.tasks])
        # self.encoder_att_4 = nn.ModuleList([self.att_layer(96 * 2, 96 // 4, 96) for _ in self.tasks])
        # self.encoder_att_5 = nn.ModuleList([self.att_layer(160 * 2, 160 // 4, 160) for _ in self.tasks])
        # self.encoder_att_6 = nn.ModuleList([self.att_layer(320 * 2, 160 // 4, 1280) for _ in self.tasks])
        # # self.encoder_att_7 = nn.ModuleList([self.att_layer(1600, 1600 // 4, 1280) for _ in self.tasks])
        # self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        # # Define task shared attention modules
        # self.encoder_block_att_0 = self.conv_layer(16, 24 // 4)
        # self.encoder_block_att_1 = self.conv_layer(24, 32 // 4)
        # self.encoder_block_att_2 = self.conv_layer(32, 64 // 4)
        # self.encoder_block_att_3 = self.conv_layer(64, 96 // 4)
        # self.encoder_block_att_4 = self.conv_layer(96, 160 // 4)
        # self.encoder_block_att_5 = self.conv_layer(160, 320 // 4)

        # Define task specific decoder
        self.num_classes = num_classes
        self.ssd_decoder = SSD(in_channels=1280, num_classes=self.num_classes)
        self.depth_decoder = Depth_decoder()

    def forward(self, x):
        ox = x
        u_0_b = self.shared_layer0_b(x)
        u_0_t = self.shared_layer0_t(u_0_b)

        # Shared MobileNet block 1
        u_1_b = self.shared_layer1_b(u_0_t)
        u_1_t = self.shared_layer1_t(u_1_b)

        # Shared MobileNet block 2
        u_2_b = self.shared_layer2_b(u_1_t)  # stride=2
        u_2_t = self.shared_layer2_t(u_2_b)

        # Shared MobileNet block 3
        u_3_b = self.shared_layer3_b(u_2_t)
        u_3_t = self.shared_layer3_t(u_3_b)

        # Shared MobileNet block 4
        u_4_b = self.shared_layer4_b(u_3_t)
        u_4_t = self.shared_layer4_t(u_4_b)

        # Shared MobileNet block 5
        u_5_b = self.shared_layer5_b(u_4_t)
        u_5_t = self.shared_layer5_t(u_5_b)

        # Shared MobileNet block 6
        u_6_b = self.shared_layer6_b(u_5_t)
        u_6_t = self.shared_layer6_t(u_6_b)

        # # Attention block 0 ->` Apply attention over last residual block
        # a_0_mask = [att_i(u_0_b) for att_i in self.encoder_att_0]  # Generate task specific attention map
        # a_0 = [a_0_mask_i * u_0_t for a_0_mask_i in a_0_mask]  # Apply task specific attention map to shared features
        # a_0 = [self.down_sampling(self.encoder_block_att_0(a_0_i)) for a_0_i in a_0]

        # # Attention block 1 ->` Apply attention over last residual block

        # a_1_mask = [att_i(torch.cat((u_1_b, a_0_i), dim=1)) for a_0_i, att_i in zip(a_0, self.encoder_att_1)]  # Generate task specific attention map
        # a_1 = [a_1_mask_i * u_1_t for a_1_mask_i in a_1_mask]  # Apply task specific attention map to shared features
        # a_1 = [self.down_sampling(self.encoder_block_att_1(a_1_i)) for a_1_i in a_1]

        # # Attention block 2 -> Apply attention over last residual block
        # a_2_mask = [att_i(torch.cat((u_2_b, a_1_i), dim=1)) for a_1_i, att_i in zip(a_1, self.encoder_att_2)]
        # a_2 = [a_2_mask_i * u_2_t for a_2_mask_i in a_2_mask]
        # a_2 = [self.down_sampling(self.encoder_block_att_2(a_2_i)) for a_2_i in a_2]

        # # Attention block 3 -> Apply attention over last residual block
        # a_3_mask = [att_i(torch.cat((u_3_b, a_2_i), dim=1)) for a_2_i, att_i in zip(a_2, self.encoder_att_3)]
        # a_3 = [a_3_mask_i * u_3_t for a_3_mask_i in a_3_mask]
        # a_3 = [self.encoder_block_att_3(a_3_i) for a_3_i in a_3]

        # # Attention block 4 -> Apply attention over last residual block (without final encoder)
        # a_4_mask = [att_i(torch.cat((u_4_b, a_3_i), dim=1)) for a_3_i, att_i in zip(a_3, self.encoder_att_4)]
        # a_4_f = [a_4_mask_i * u_4_t for a_4_mask_i in a_4_mask]
        # a_4 = [self.down_sampling(self.encoder_block_att_4(a_4_i)) for a_4_i in a_4_f]

        # # # Attention block 5 -> Apply attention over last residual block (without final encoder)
        # a_5_mask = [att_i(torch.cat((u_5_b, a_4_i), dim=1)) for a_4_i, att_i in zip(a_4, self.encoder_att_5)]
        # a_5 = [a_5_mask_i * u_5_t for a_5_mask_i in a_5_mask]
        # a_5 = [self.encoder_block_att_5(a_5_i) for a_5_i in a_5]

        # # # Attention block 6 -> Apply attention over last residual block (without final encoder)
        # a_6_mask = [att_i(torch.cat((u_6_b, a_5_i), dim=1)) for a_5_i, att_i in zip(a_5, self.encoder_att_6)]
        # a_6 = [a_6_mask_i * u_6_t for a_6_mask_i in a_6_mask]

        # task1 specific decoder
        source = list()
        loc = list()
        conf = list()

        # the output from conv4_3 -> shared_layer4_t
        source.append(self.ssd_decoder.L2Norm(u_4_t))
        # the output from conv7
        source.append(u_6_t)
        # extract the output from the ssd layer
        ssd_forward = u_6_t

        for v in self.ssd_decoder.extras:
            ssd_forward = F.relu(v(ssd_forward), inplace=True)
            source.append(ssd_forward)

        for (x, l, c) in zip(source, self.ssd_decoder.loc, self.ssd_decoder.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
            # print(x.size())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        # task2 specific decoder
        depth = self.depth_decoder(ox, u_6_t)
        output = (  # localization
            loc.view(loc.size(0), -1, 4),
            # classification
            conf.view(conf.size(0), -1, self.num_classes)
        )

        return [output, depth], self.logsigma

    def block_mobilenet(self):
        mobile_modules = [
            self.shared_layer0_b,
            self.shared_layer0_t,
            self.shared_layer1_b,
            self.shared_layer1_t,
            self.shared_layer2_b,
            self.shared_layer2_t,
            self.shared_layer3_b,
            self.shared_layer3_t,
            self.shared_layer4_b,
            self.shared_layer4_t,
            self.shared_layer5_b,
            self.shared_layer5_t,
            self.shared_layer6_b,
            self.shared_layer6_t]
        print("block the mobile net")
        for mm in mobile_modules:
            for p in mm.parameters():
                p.requires_grad = False

    def unblock_mobilenet(self):
        mobile_modules = [
            self.shared_layer0_b,
            self.shared_layer0_t,
            self.shared_layer1_b,
            self.shared_layer1_t,
            self.shared_layer2_b,
            self.shared_layer2_t,
            self.shared_layer3_b,
            self.shared_layer3_t,
            self.shared_layer4_b,
            self.shared_layer4_t,
            self.shared_layer5_b,
            self.shared_layer5_t,
            self.shared_layer6_b,
            self.shared_layer6_t]
        print("Unblock the mobile net")
        for mm in mobile_modules:
            for p in mm.parameters():
                p.requires_grad = True

    def focous_on_ssd(self):
        other = [
            self.shared_layer0_b,
            self.shared_layer0_t,
            self.shared_layer1_b,
            self.shared_layer1_t,
            self.shared_layer2_b,
            self.shared_layer2_t,
            self.shared_layer3_b,
            self.shared_layer3_t,
            self.shared_layer4_b,
            self.shared_layer4_t,
            self.shared_layer5_b,
            self.shared_layer5_t,
            self.shared_layer6_b,
            self.shared_layer6_t,
            self.encoder_block_att_0,
            self.encoder_block_att_1,
            self.encoder_block_att_2,
            self.encoder_block_att_3,
            self.encoder_block_att_4,
            self.encoder_block_att_5,
            self.encoder_att_0[1],
            self.encoder_att_1[1],
            self.encoder_att_2[1],
            self.encoder_att_3[1],
            self.encoder_att_4[1],
            self.encoder_att_5[1],
            self.depth_decoder,
        ]
        print("focous on ssd.")

        for mm in other:
            for p in mm.parameters():
                p.requires_grad = False

    def att_layer(self, in_channel, hidden_channel, out_channel):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channel,
                      out_channels=hidden_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(hidden_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_channel,
                      out_channels=out_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channel),
            nn.Sigmoid()
        )

    def conv_layer(self, in_channel, out_channel, stride=1):
        downsample = nn.Sequential(conv1x1(in_channel, 4 * out_channel, stride=1),
                                   nn.BatchNorm2d(4 * out_channel))
        return Bottleneck(in_channel, out_channel, stride=stride, downsample=downsample)
