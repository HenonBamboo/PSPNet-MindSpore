import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.ops.operations as P
import src.model.resnet as models
from src.model.resnet import bn


class PPM(nn.Cell):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        bins = [60, 30, 20, 10]
        for bin in bins:
            self.features.append(nn.SequentialCell([
                # AdaptiveAvgPool2d(bin),
                nn.AvgPool2d(bin, bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, pad_mode='pad', has_bias=False),
                bn(reduction_dim),
                nn.ReLU()
            ]))
        self.features = nn.CellList(self.features)
        self.resize = nn.ResizeBilinear()

    def construct(self, x):
        h, w = P.Shape()(x)[-2:]
        out = [x]
        for f in self.features:
            out.append(self.resize(f(x), size=(h, w), align_corners=True))
            # out.append(ops.interpolate(f(x), sizes=(h, w), mode='bilinear', coordinate_transformation_mode="align_corners"))
        return P.Concat(1)(out)


class PSPNet(nn.Cell):
    def __init__(self, backbone="resnet50", pool_sizes=(1, 2, 3, 6), num_classes=21, pretrained=False,
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

        fea_dim = 2048

        self.ppm = PPM(fea_dim, int(fea_dim/len(pool_sizes)), pool_sizes)
        fea_dim *= 2

        self.cls = nn.SequentialCell([
            nn.Conv2d(fea_dim, 512, kernel_size=3, pad_mode='pad', padding=1, has_bias=False),
            bn(512),
            nn.ReLU(),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, pad_mode='pad', has_bias=True)
        ])
        self.aux = nn.SequentialCell([
            nn.Conv2d(1024, 256, kernel_size=3, pad_mode='pad', padding=1, has_bias=False),
            bn(256),
            nn.ReLU(),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, pad_mode='pad', has_bias=True)
        ])
        self.resize = nn.ResizeBilinear()

    def construct(self, x):
        h, w = P.Shape()(x)[-2:]
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_tmp = self.layer3(x)
        x = self.layer4(x_tmp)
        x = self.ppm(x)
        x = self.cls(x)
        x = self.resize(x, size=(h, w), align_corners=True)
        # x = ops.interpolate(x, sizes=(h, w), mode='bilinear', coordinate_transformation_mode="align_corners")
        return x


if __name__ == '__main__':
    from mindspore import context
    context.set_context(mode=context.GRAPH_MODE)

    x = ops.rand(4, 3, 473, 473)
    model = PSPNet()
    y = model(x)
    print(y.shape)
