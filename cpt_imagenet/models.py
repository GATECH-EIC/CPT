import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import torch.autograd as autograd
from modules.quantize import QConv2d
import torch.nn.functional as F



def Conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return QConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                   padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1, padding=0):
    return QConv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                   padding=padding, bias=False)

def conv(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
    return QConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                   padding=padding, dilation=dilation, groups=groups, bias=bias)

def make_bn(planes):
	return nn.BatchNorm2d(planes)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, num_bits, num_grad_bits):
        residual = x

        out = self.conv1(x, num_bits, num_grad_bits)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out, num_bits, num_grad_bits)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * 4)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, num_bits, num_grad_bits):
        residual = x

        out = self.conv1(x, num_bits, num_grad_bits)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out, num_bits, num_grad_bits)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out, num_bits, num_grad_bits)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.num_layers = layers

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_group(block, 64, layers[0], group_id=1)
        self.layer2 = self._make_group(block, 128, layers[1], group_id=2)
        self.layer3 = self._make_group(block, 256, layers[2], group_id=3)
        self.layer4 = self._make_group(block, 512, layers[3], group_id=4)

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            if isinstance(m, QConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(0) * m.weight.size(1)
                m.weight.data.normal_(0, math.sqrt(2. / n))


    def _make_group(self, block, planes, layers, group_id):
        """ Create the whole group"""
        for i in range(layers):
            if group_id > 1 and i == 0:
                stride = 2
            else:
                stride = 1

            layer = self._make_layer(block, planes, stride=stride)

            setattr(self, 'group{}_layer{}'.format(group_id, i), layer)


    def _make_layer(self, block, planes, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layer = block(self.inplanes, planes, stride, downsample)
        self.inplanes = planes * block.expansion

        return layer

    def forward(self, x, num_bits, num_grad_bits):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for g in range(len(self.num_layers)):
            for i in range(self.num_layers[g]):
                x = getattr(self, 'group{}_layer{}'.format(g+1, i))(x, num_bits, num_grad_bits)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def resnet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model









