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


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
    return QConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                   padding=padding, dilation=dilation, groups=groups, bias=bias)


def make_bn(planes):
	return nn.BatchNorm2d(planes)
	# return RangeBN(planes)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = make_bn(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = make_bn(planes)
        self.bn3 = make_bn(planes)
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
            residual = self.downsample(x, num_bits, num_grad_bits)
            residual = self.bn3(residual)

        out  += residual
        out = self.relu(out)
        return out


########################################
# Original ResNet                      #
########################################

class ResNet(nn.Module):
    """Original ResNet without routing modules"""
    def __init__(self, block, layers, num_classes=10):
        self.inplanes = 16
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 16)
        self.bn1 = make_bn(16)
        self.relu = nn.ReLU(inplace=True)

        self.num_layers = layers

        self._make_group(block, 16, layers[0], group_id=1,
                         )
        self._make_group(block, 32, layers[1], group_id=2,
                         )
        self._make_group(block, 64, layers[2], group_id=3,
                         )

        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(0) * m.weight.size(1)
                m.weight.data.normal_(0, math.sqrt(2. / n))


    def _make_group(self, block, planes, layers, group_id=1
                    ):
        """ Create the whole group"""
        for i in range(layers):
            if group_id > 1 and i == 0:
                stride = 2
            else:
                stride = 1

            layer = self._make_layer_v2(block, planes, stride=stride,
                                       )
            setattr(self, 'group{}_layer{}'.format(group_id, i), layer)


    def _make_layer_v2(self, block, planes, stride=1,
                       ):
        """ create one block and optional a gate module """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = QConv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)

        layer = block(self.inplanes, planes, stride, downsample)
        self.inplanes = planes * block.expansion

        return layer

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = QConv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x, num_bits, num_grad_bits):
        x = self.conv1(x, num_bits, num_grad_bits)
        x = self.bn1(x)
        x = self.relu(x)

        for g in range(3):
            for i in range(self.num_layers[g]):
                x = getattr(self, 'group{}_layer{}'.format(g+1, i))(x, num_bits, num_grad_bits)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# For CIFAR-10
# ResNet-38
def cifar10_resnet_38(pretrained=False, **kwargs):
    # n = 6
    model = ResNet(BasicBlock, [6, 6, 6], **kwargs)
    return model


# ResNet-74
def cifar10_resnet_74(pretrained=False, **kwargs):
    # n = 12
    model = ResNet(BasicBlock, [12, 12, 12], **kwargs)
    return model


# ResNet-110
def cifar10_resnet_110(pretrained=False, **kwargs):
    # n = 18
    model = ResNet(BasicBlock, [18, 18, 18], **kwargs)
    return model


# ResNet-152
def cifar10_resnet_152(pretrained=False, **kwargs):
    # n = 25
    model = ResNet(BasicBlock, [25, 25, 25], **kwargs)
    return model


# For CIFAR-100
# ResNet-38
def cifar100_resnet_38(pretrained=False, **kwargs):
    # n = 6
    model = ResNet(BasicBlock, [6, 6, 6], num_classes=100)
    return model


# ResNet-74
def cifar100_resnet_74(pretrained=False, **kwargs):
    # n = 12
    model = ResNet(BasicBlock, [12, 12, 12], num_classes=100)
    return model


# ResNet-110
def cifar100_resnet_110(pretrained=False, **kwargs):
    # n = 18
    model = ResNet(BasicBlock, [18, 18, 18], num_classes=100)
    return model


# ResNet-152
def cifar100_resnet_152(pretrained=False, **kwargs):
    # n = 25
    model = ResNet(BasicBlock, [25, 25, 25], num_classes=100)
    return model



class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = conv(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = None
        if stride == 1 and in_planes != out_planes:
            self.shortcut = conv(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn4 = nn.BatchNorm2d(out_planes)


    def forward(self, x, num_bits, num_grad_bits):
        out = F.relu(self.bn1(self.conv1(x, num_bits, num_grad_bits)))
        out = F.relu(self.bn2(self.conv2(out, DWS_BITS, DWS_GRAD_BITS)))
        out = self.bn3(self.conv3(out, num_bits, num_grad_bits))

        if self.stride == 1:
            if self.shortcut:
                out = out + self.bn4(self.shortcut(x, num_bits, num_grad_bits))
            else:
                out = out + x
        return out


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = conv(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self._make_layers(in_planes=32)
        self.conv2 = conv(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

        self.num_layers = [item[2] for item in self.cfg]

    def _make_layers(self, in_planes):

        for i, (expansion, out_planes, num_blocks, stride) in enumerate(self.cfg):
            strides = [stride] + [1]*(num_blocks-1)

            for j, stride in enumerate(strides):
                setattr(self, 'group{}_layer{}'.format(i+1, j), Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes


    def forward(self, x, num_bits, num_grad_bits):
        x = F.relu(self.bn1(self.conv1(x, num_bits, num_grad_bits)))

        for g in range(7):
            for i in range(self.num_layers[g]):
                x = getattr(self, 'group{}_layer{}'.format(g+1, i))(x, num_bits, num_grad_bits)

        x = F.relu(self.bn2(self.conv2(x, num_bits, num_grad_bits)))

        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


def cifar10_mobilenet_v2(pretrained=False, **kwargs):
    return MobileNetV2(num_classes=10, **kwargs)


def cifar100_mobilenet_v2(pretrained=False, **kwargs):
    return MobileNetV2(num_classes=100, **kwargs)