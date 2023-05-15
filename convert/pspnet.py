import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.ops.operations as P

import convert.resnet as models


class PPM(nn.Cell):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.SequentialCell([
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, pad_mode='pad', has_bias=False),
                nn.BatchNorm2d(reduction_dim, momentum=0.9),
                nn.ReLU()
            ]))
        self.features = nn.CellList(self.features)

    def construct(self, x):
        x = self.cast(x, ms.float32)
        x_size = P.Shape()(x)
        out = [x]
        for f in self.features:
            out.append(ops.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return P.Concat(1)(out)


class PSPNet(nn.Cell):
    def __init__(self, backbone="resnet50", pool_sizes=(1, 2, 3, 6), num_classes=2, pretrained=False, aux_branch=True,
                 pretrained_path="", deep_base=True):
        super(PSPNet, self).__init__()
        assert 2048 % len(pool_sizes) == 0
        assert num_classes > 1

        if backbone == "resnet50":
            resnet = models.resnet50(pretrained=pretrained, deep_base=deep_base, pretrained_path=pretrained_path)
        elif backbone == "resnet101":
            resnet = models.resnet101(pretrained=pretrained, deep_base=deep_base, pretrained_path=pretrained_path)
        else:
            resnet = models.resnet152(pretrained=pretrained, deep_base=deep_base, pretrained_path=pretrained_path)
        self.layer0 = nn.SequentialCell([resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu,
                                         resnet.conv3, resnet.bn3, resnet.relu, resnet.maxpool])
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.cells_and_names():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.cells_and_names():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        fea_dim = 2048

        self.ppm = PPM(fea_dim, int(fea_dim/len(pool_sizes)), pool_sizes)
        fea_dim *= 2

        self.cls = nn.SequentialCell([
            nn.Conv2d(fea_dim, 512, kernel_size=3, pad_mode='pad', padding=1, has_bias=False),
            nn.BatchNorm2d(512, momentum=0.9),
            nn.ReLU(),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, pad_mode='pad', has_bias=True)
        ])

        self.aux_branch = aux_branch
        if self.aux_branch:
            self.aux = nn.SequentialCell([
                nn.Conv2d(1024, 256, kernel_size=3, pad_mode='pad', padding=1, has_bias=False),
                nn.BatchNorm2d(256, momentum=0.9),
                nn.ReLU(),
                nn.Dropout2d(p=0.1),
                nn.Conv2d(256, num_classes, kernel_size=1, pad_mode='pad', has_bias=True)
            ])

    def construct(self, x):
        x_shape = P.Shape()(x)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_tmp = self.layer3(x)
        x = self.layer4(x_tmp)
        x = self.ppm(x)
        x = self.cls(x)
        x = ops.interpolate(x, size=(x_shape[2:4]), mode='bilinear', align_corners=True)

        if self.aux_branch:
            aux = self.aux(x_tmp)
            aux = ops.interpolate(aux, size=(x_shape[2:4]), mode='bilinear', align_corners=True)
            return aux, x
        else:
            return x


if __name__ == '__main__':
    from mindspore import context
    from mindspore.train.serialization import load_checkpoint, load_param_into_net

    context.set_context(mode=context.PYNATIVE_MODE)
    # context.set_context(mode=context.GRAPH_MODE)

    input = ops.rand(4, 3, 473, 473)
    model = PSPNet(num_classes=21, pretrained=False)
    y = model(input)
    print(y.shape)
    # for i in model.trainable_params():
    #     print(i)

    # params_dict = {}
    # for name, param in model.parameters_and_names():
    #     params_dict[name] = param.name
    # print(params_dict)
    # print('Begin load pretrained model!')
    # param_dict = load_checkpoint('weight_train_epoch_50.ckpt')
    # for key, value in param_dict.copy().items():
    #     if 'head' in key:
    #         if value.shape[0] != 256:
    #             print(f'==> removing {key} with shape {value.shape}')
    #             param_dict.pop(key)
    # new_param_dict = {}
    # for key, value in param_dict.copy().items():
    #     new_param_dict[params_dict[key]] = value
    # load_param_into_net(model, new_param_dict)
    # print('Load pretrained model success!')
    #
    # output = model(input)
    # print('PSPNet', output.shape)

