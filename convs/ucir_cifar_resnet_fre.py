'''
Reference:
https://github.com/khurramjaved96/incremental-learning/blob/autoencoders/model/resnet32.py
https://github.com/hshustc/CVPR19_Incremental_Learning/blob/master/cifar100-class-incremental/modified_resnet_cifar.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
from convs.cifar_resnet_decoder import *


# from convs.modified_linear import CosineLinear
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class DownsampleA(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleA, self).__init__()
        assert stride == 2
        self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.avg(x)
        return torch.cat((x, x.mul(0)), 1)


class DownsampleB(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleB, self).__init__()
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(nOut)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class DownsampleC(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleC, self).__init__()
        assert stride != 1 or nIn != nOut
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        x = self.conv(x)
        return x


class DownsampleD(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleD, self).__init__()
        assert stride == 2
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=2, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(nOut)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class ResNetBasicblock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, last=False):
        super(ResNetBasicblock, self).__init__()

        self.conv_a = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_a = nn.BatchNorm2d(planes)

        self.conv_b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_b = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.last = last

    def forward(self, x):
        residual = x

        basicblock = self.conv_a(x)
        basicblock = self.bn_a(basicblock)
        basicblock = F.relu(basicblock, inplace=True)

        basicblock = self.conv_b(basicblock)
        basicblock = self.bn_b(basicblock)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = residual + basicblock
        if not self.last:
            out = F.relu(out, inplace=True)

        return out


class CifarResNet(nn.Module):
    """
    ResNet optimized for the Cifar Dataset, as specified in
    https://arxiv.org/abs/1512.03385.pdf
    """

    def __init__(self, block, depth, channels=3):
        super(CifarResNet, self).__init__()

        # Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
        assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
        layer_blocks = (depth - 2) // 6

        self.conv_1_3x3 = nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(16)

        self.inplanes = 16
        self.stage_1 = self._make_layer(block, 16, layer_blocks, 1)
        self.stage_2 = self._make_layer(block, 32, layer_blocks, 2)
        self.stage_3 = self._make_layer(block, 64, layer_blocks, 2, last_phase=True)
        self.avgpool = nn.AvgPool2d(8)
        self.out_dim = 64 * block.expansion
        # self.fc = CosineLinear(64*block.expansion, 10)

        # add high and low filter branch
        self.distangler_H = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                          nn.BatchNorm2d(64),
                                          nn.ReLU(),
                                          )

        self.distangler_L = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                          nn.BatchNorm2d(64),
                                          nn.ReLU(),
                                          )

        self.conv1x1 = conv1x1(64, 64)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # add decoder subnetwork
        self.decoder_H = resnet18_decoder()
        self.decoder_L = resnet18_decoder()

    def _make_layer(self, block, planes, blocks, stride=1, last_phase=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleB(self.inplanes, planes * block.expansion, stride)  # DownsampleA => DownsampleB

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        if last_phase:
            for i in range(1, blocks-1):
                layers.append(block(self.inplanes, planes))
            layers.append(block(self.inplanes, planes, last=True))
        else:
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, phase='test'):
        x = self.conv_1_3x3(x)  # [bs, 16, 32, 32]
        x = F.relu(self.bn_1(x), inplace=True)

        x_1 = self.stage_1(x)  # [bs, 16, 32, 32]
        x_2 = self.stage_2(x_1)  # [bs, 32, 16, 16]
        x_3 = self.stage_3(x_2)  # [bs, 64, 8, 8]

        if phase == 'train':
            L_info = self.distangler_L(x_3)
            x_L_image = self.decoder_L(L_info)
            H_info = self.distangler_H(x_3)
            x_H_image = self.decoder_H(H_info)
            features = self.conv1x1(L_info) + H_info
            features = self.avgpool(features)
            features = features.view(features.size(0), -1)

            return {
                'fmaps': [x_1, x_2, x_3],
                'features': features,
                'reconstruct_L': x_L_image,
                'reconstruct_H': x_H_image
            }
        elif phase == 'test':
            L_info = self.distangler_L(x_3)
            H_info = self.distangler_H(x_3)

            # features = torch.cat((L_info, H_info), dim=1)
            # features = L_info + H_info
            features = self.conv1x1(L_info) + H_info
            features = self.avgpool(features)
            features = features.view(features.size(0), -1)

            return {
                'fmaps': [x_1, x_2, x_3],
                'features': features,
                'reconstruct_L': None,
                'reconstruct_H': None
            }

    @property
    def last_conv(self):
        return self.stage_3[-1].conv_b


def resnet20mnist_fre():
    """Constructs a ResNet-20 model for MNIST."""
    model = CifarResNet(ResNetBasicblock, 20, 1)
    return model


def resnet32mnist_fre():
    """Constructs a ResNet-32 model for MNIST."""
    model = CifarResNet(ResNetBasicblock, 32, 1)
    return model


def resnet20_fre():
    """Constructs a ResNet-20 model for CIFAR-10."""
    model = CifarResNet(ResNetBasicblock, 20)
    return model


def resnet32_fre():
    """Constructs a ResNet-32 model for CIFAR-10."""
    model = CifarResNet(ResNetBasicblock, 32)
    return model


def resnet44_fre():
    """Constructs a ResNet-44 model for CIFAR-10."""
    model = CifarResNet(ResNetBasicblock, 44)
    return model


def resnet56_fre():
    """Constructs a ResNet-56 model for CIFAR-10."""
    model = CifarResNet(ResNetBasicblock, 56)
    return model


def resnet110_fre():
    """Constructs a ResNet-110 model for CIFAR-10."""
    model = CifarResNet(ResNetBasicblock, 110)
    return model


