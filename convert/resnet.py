import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore.train.serialization import load_checkpoint, load_param_into_net


__all__ = ['resnet50', 'resnet101', 'resnet152']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, pad_mode='pad', padding=1, has_bias=False)


class BasicBlock(nn.Cell):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.9)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.9)
        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Cell):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, pad_mode='pad', has_bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.9)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, pad_mode='pad', padding=1, has_bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.9)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, pad_mode='pad', has_bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=0.9)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Cell):

    def __init__(self, block, layers, num_classes=1000, deep_base=True):
        super(ResNet, self).__init__()
        self.deep_base = deep_base
        if not self.deep_base:
            self.inplanes = 64
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, pad_mode='pad', padding=3,
                                   has_bias=False)
            self.bn1 = nn.BatchNorm2d(num_features=64, momentum=0.9)
        else:
            self.inplanes = 128
            self.conv1 = conv3x3(3, 64, stride=2)
            self.bn1 = nn.BatchNorm2d(64, momentum=0.9)
            self.conv2 = conv3x3(64, 64)
            self.bn2 = nn.BatchNorm2d(64, momentum=0.9)
            self.conv3 = conv3x3(64, 128)
            self.bn3 = nn.BatchNorm2d(128, momentum=0.9)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1, pad_mode='valid')
        self.fc = nn.Dense(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell([
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, pad_mode='pad',
                          has_bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=0.9),
            ])

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.SequentialCell([*layers])

    def construct(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        if self.deep_base:
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = P.Reshape()(x, (P.Shape()(x)[0], -1))
        x = self.fc(x)

        return x


def resnet50(pretrained=False, pretrained_path="", deep_base=True):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], deep_base=deep_base)

    if pretrained:
        params_dict = {}
        for name, param in model.parameters_and_names():
            params_dict[name] = param.name
        print(params_dict)
        print('Begin load pretrained model!')
        param_dict = load_checkpoint(pretrained_path)
        for key, value in param_dict.copy().items():
            if 'head' in key:
                if value.shape[0] != 256:
                    print(f'==> removing {key} with shape {value.shape}')
                    param_dict.pop(key)
        new_param_dict = {}
        for key, value in param_dict.copy().items():
            new_param_dict[params_dict[key]] = value
        load_param_into_net(model, new_param_dict)
        print('Load pretrained model success!')
    return model


def resnet101(pretrained=False, pretrained_path="", deep_base=True):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], deep_base=deep_base)

    if pretrained:
        params_dict = {}
        for name, param in model.parameters_and_names():
            params_dict[name] = param.name
        print(params_dict)
        print('Begin load pretrained model!')
        param_dict = load_checkpoint(pretrained_path)
        for key, value in param_dict.copy().items():
            if 'head' in key:
                if value.shape[0] != 256:
                    print(f'==> removing {key} with shape {value.shape}')
                    param_dict.pop(key)
        new_param_dict = {}
        for key, value in param_dict.copy().items():
            new_param_dict[params_dict[key]] = value
        load_param_into_net(model, new_param_dict)
        print('Load pretrained model success!')
    return model


def resnet152(pretrained=False, pretrained_path="", deep_base=True):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], deep_base=deep_base)

    if pretrained:
        params_dict = {}
        for name, param in model.parameters_and_names():
            params_dict[name] = param.name
        print(params_dict)
        print('Begin load pretrained model!')
        param_dict = load_checkpoint(pretrained_path)
        for key, value in param_dict.copy().items():
            if 'head' in key:
                if value.shape[0] != 256:
                    print(f'==> removing {key} with shape {value.shape}')
                    param_dict.pop(key)
        new_param_dict = {}
        for key, value in param_dict.copy().items():
            new_param_dict[params_dict[key]] = value
        load_param_into_net(model, new_param_dict)
        print('Load pretrained model success!')
    return model
