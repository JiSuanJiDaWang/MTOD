import torch
import torch.nn as nn
import math
import torch.nn.functional as F



class SELayer(nn.Module):
    """
    Taken from:
    https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4
    """
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        #self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = torch.mean(x, dim=[2,3]) # Replacement of avgPool for large kernels for trt
        y = y.view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand(x.shape)



class Guided_Upsampling_Block(nn.Module):
    def __init__(self, in_features, expand_features, out_features,
                 kernel_size=3, channel_attention=True,
                 guidance_type='full', guide_features=3):
        super(Guided_Upsampling_Block, self).__init__()

        self.channel_attention = channel_attention
        self.guidance_type = guidance_type
        self.guide_features = guide_features
        self.in_features = in_features

        padding = kernel_size // 2

        self.feature_conv = nn.Sequential(
            nn.Conv2d(in_features, expand_features,
                      kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(expand_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(expand_features, expand_features // 2, kernel_size=1),
            nn.BatchNorm2d(expand_features // 2),
            nn.ReLU(inplace=True))

        if self.guidance_type == 'full':
            self.guide_conv = nn.Sequential(
                nn.Conv2d(self.guide_features, expand_features,
                          kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(expand_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(expand_features, expand_features // 2, kernel_size=1),
                nn.BatchNorm2d(expand_features // 2),
                nn.ReLU(inplace=True))

            comb_features = (expand_features // 2) * 2
        elif self.guidance_type =='raw':
            comb_features = expand_features // 2 + guide_features
        else:
            comb_features = expand_features // 2

        self.comb_conv = nn.Sequential(
            nn.Conv2d(comb_features, expand_features,
                      kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(expand_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(expand_features, in_features, kernel_size=1),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True))

        self.reduce = nn.Conv2d(in_features,
                                out_features,
                                kernel_size=1)

        if self.channel_attention:
            self.SE_block = SELayer(comb_features,
                                    reduction=1)


    def forward(self, guide, depth):
        x = self.feature_conv(depth)

        if self.guidance_type == 'full':
            y = self.guide_conv(guide)
            xy = torch.cat([x, y], dim=1)
        elif self.guidance_type == 'raw':
            xy = torch.cat([x, guide], dim=1)
        else:
            xy = x

        if self.channel_attention:
            xy = self.SE_block(xy)

        residual = self.comb_conv(xy)
        return self.reduce(residual + depth)


import torch
import torch.nn as nn
import torch.nn.functional as F



class GuideDepth(nn.Module):
    def __init__(self, 
            up_features=[64, 32, 16], 
            inner_features=[64, 32, 16]):
        super(GuideDepth, self).__init__()



        self.up_1 = Guided_Upsampling_Block(in_features=up_features[0],
                                   expand_features=inner_features[0],
                                   out_features=up_features[1],
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=3,
                                   guidance_type="full")
        self.up_2 = Guided_Upsampling_Block(in_features=up_features[1],
                                   expand_features=inner_features[1],
                                   out_features=up_features[2],
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=3,
                                   guidance_type="full")
        self.up_3 = Guided_Upsampling_Block(in_features=up_features[2],
                                   expand_features=inner_features[2],
                                   out_features=1,
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=3,
                                   guidance_type="full")


    def forward(self, x, y):
        # y is the output of the decoder.

        x_half = F.interpolate(x, scale_factor=.5)
        x_quarter = F.interpolate(x, scale_factor=.25)

        y = F.interpolate(y, scale_factor=2, mode='bilinear')
        y = self.up_1(x_quarter, y)

        y = F.interpolate(y, scale_factor=2, mode='bilinear')
        y = self.up_2(x_half, y)

        y = F.interpolate(y, scale_factor=2, mode='bilinear')
        y = self.up_3(x, y)
        return y
# Define Fast-depth decoder
def depthwise(in_channels, kernel_size):
    padding = (kernel_size-1) // 2
    assert 2*padding == kernel_size-1, "parameters incorrect. kernel={}, padding={}".format(kernel_size, padding)
    return nn.Sequential(
          nn.Conv2d(in_channels, in_channels, kernel_size, stride=1, padding=padding, bias=False, groups=in_channels),
          nn.BatchNorm2d(in_channels),
          nn.ReLU(inplace=True),
        )


def pointwise(in_channels, out_channels):
    return nn.Sequential(
          nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
          nn.BatchNorm2d(out_channels),
          nn.ReLU(inplace=True),
        )


class FbNet_block(nn.Module):

    def __init__(self, in_channels, out_channels, expand_rate=3, Upsampling=True):
        super(FbNet_block, self).__init__()

        self.conv1 = pointwise(in_channels, in_channels*expand_rate)
        self.conv2 = depthwise(in_channels*expand_rate, kernel_size=5)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels*expand_rate,out_channels, 1, 1, 0,bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.near = nn.UpsamplingNearest2d(scale_factor=2)
        self.Upsampling = Upsampling
    def forward(self, x):
        x = self.conv1(x)
        if self.Upsampling:
            x = self.near(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return x

class Depth_decoder(nn.Module):


    def __init__(self, kernel_size=5):
        super(Depth_decoder, self).__init__()

        self.adjust_conv = pointwise(1280,256)
        self.conv1 = nn.Sequential(
            depthwise(256, kernel_size),
            pointwise(256, 128)
        )
        self.conv2 = nn.Sequential(
            depthwise(128, kernel_size),
            pointwise(128, 64)
        )
        self.guided = GuideDepth()

        weights_init(self.adjust_conv)
        weights_init(self.conv1)
        weights_init(self.conv2)
        weights_init(self.guided)


    def forward(self, ox, x):

        x = self.adjust_conv(x)
        x = self.conv1(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')  # 14*14
        x = self.conv2(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')  # 28*28
        x = self.guided(ox, x)


        return x


def weights_init(m):
    # Initialize kernel weights with Gaussian distributions
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

