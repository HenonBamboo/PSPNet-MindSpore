# PSPNet 网络

- PSPNet（Pyramid Scene Parsing Network，金字塔场景解析网络）通过金字塔池模块将不同区域的上下文聚合在一起，具有强大的全局上下文信息能力。

- [paper](https://arxiv.org/abs/1612.01105) from CVPR2017

- 金字塔池模块融合了四种不同金字塔尺度下的特征。为了保持合理的表示间隙，该模块是一个四级模块，bin大小分别为1×1、2×2、3×3和6×6。

# 数据集

- [VOC2012AUG](https://zhuanlan.zhihu.com/p/158622375)

- SBD是VOC2011和VOC2012的一个增强版本，但是他们之间的兼容性还存在一些问题：
（1） VOC2012 train set中有331张image，不包含在 SBD train set中。这导致了，无论 VOC2012 还是 SBD 的 train set 都不是最强的。
（2） VOC2012 val set中有545张image，包含在 SBD train set中。这导致了，使用SBD train set作为训练集、VOC2012作为验证集时，只能使用VOC2012的subval set。

- 考虑到以上两点不足，PASCAL VOC 2012 Augment（VOC2012AUG）就诞生了。 + 针对问题（2），VOC2012AUG 的 val set 就等于 VOC2012 的 val set + 针对问题（1），VOC2012AUG 的 train set 聚合了 VOC2012 train set(a)、SBD train set(b) 和 SBD val set(c)，并扣除了 b与a重复的部分、c与a重复的部分、b与VOC2012 val set重复的部分、c与VOC2012 val set重复的部分。

- 数据集构成：

- PASCAL VOC2012AUG 的构成如下：

- train：10582
- val：1449
- train+val：12031

# 预训练模型

- [resnet50-imagenet pretrained model](https://drive.google.com/open?id=15wx9vOM0euyizq-M1uINgN0_wjVRf9J3)

# 环境设置

- Hardware :(Ascend)
    - Prepare ascend processor to build hardware environment
- frame:
    - [Mindspore](https://www.mindspore.cn/install)
- For details, please refer to the following resources:
    - [MindSpore course](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 脚本描述

## Script and Sample Code

```python
.
└─PSPNet
├── convert                                    # convert
├── inference                                  # inference
├── eval.py                                    # Evaluation python file for /VOC2012AUG
├── README.md                                  # descriptions about PSPNet
├── config                                     # the training config file
│   ├── ade20k_pspnet50.yaml
│   └── voc2012_pspnet50.yaml
├── src                                        # PSPNet
│   ├── dataset                                # data processing
│   │   ├── pt_dataset.py
│   │   └── pt_transform.py
│   ├── model                                  # models for training and test
│   │   ├── PSPNet.py
│   │   ├── resnet.py
│   │   └── cell.py                            # loss function
│   └── utils
│       ├── functions_args.py                  # test helper
│       ├── lr.py                              # learning rate
│       ├── metric_and_evalcallback.py         # evalcallback
│       ├── aux_loss.py                        # loss function helper
│       └── p_util.py                          # some functions                          
└── train.py                                   # The training python file for ADE20K/VOC2012-SBD
```

## 脚本参数

- Set script parameters in ./config/voc2012_pspnet50.yaml

### 模型

```bash
name: "PSPNet"
backbone: "resnet50_v2"
base_size: 512   # based size for scaling
crop_size: 473
```

### 优化器

```bash
init_lr: 0.01
momentum: 0.9
weight_decay: 0.0001
```

### 训练

```bash
batch_size: 16    # batch size for training
batch_size_val: 16  # batch size for validation during training
epochs: 50 
```

## 训练过程

### 训练

```shell
python train.py
```

### 训练结果

- 培训结果将保存在PSPNet路径中。

```bash
# training result(1p)-VOC2012AUG
Train epoch time: 215749.452 ms, per step time: 1307.572 ms
Train epoch time: 217972.430 ms, per step time: 1321.045 ms
Train epoch time: 213208.878 ms, per step time: 1292.175 ms
Train epoch time: 218587.545 ms, per step time: 1324.773 ms
Epoch: 49, val_loss: 0.31603706
Epoch: 49, val_loss: 0.242274
Epoch: 49, val_loss: 0.21327616
Epoch: 49, val_loss: 0.25860044
=== epoch:   49, device id:  3, best miou: 0.6963, miou: 0.6825
=== epoch:   49, device id:  0, best miou: 0.6969, miou: 0.6932
=== epoch:   49, device id:  2, best miou: 0.6761, miou: 0.6761
=== epoch:   49, device id:  1, best miou: 0.7048, miou: 0.6969
Train epoch time: 213211.140 ms, per step time: 1292.189 ms
Train epoch time: 213662.691 ms, per step time: 1294.925 ms
Train epoch time: 214910.849 ms, per step time: 1302.490 ms
Train epoch time: 218207.310 ms, per step time: 1322.469 ms
Epoch: 50, val_loss: 0.31445614
Epoch: 50, val_loss: 0.21240959
Epoch: 50, val_loss: 0.24067509
Epoch: 50, val_loss: 0.2589045
=== epoch:   50, device id:  2, best miou: 0.6772, miou: 0.6772
=== epoch:   50, device id:  1, best miou: 0.7048, miou: 0.6990
=== epoch:   50, device id:  3, best miou: 0.6963, miou: 0.6842
=== epoch:   50, device id:  0, best miou: 0.6969, miou: 0.6930
```

## 评估过程

### 评估

```shell
python eval.py
```

### 评估结果

```bash
VOC2012-AUG-ss:mIoU/mAcc/allAcc 0.6929/0.8305/0.9232.
````

# 型号说明

|Parameter              | PSPNet                                                   |
| ------------------- | --------------------------------------------------------- |
|resources              | Ascend 910；CPU 2.60GHz, 192core；memory：755G |
|Upload date            |2023.05.15                    |
|mindspore version      |mindspore2.0.0     |
|training parameter     |epoch=50,batch_size=16   |
|optimizer              |SGD optimizer，momentum=0.9,weight_decay=0.0001    |
|loss function          |SoftmaxCrossEntropyLoss   |
|Blog URL               |https://zhuanlan.zhihu.com/p/628391472|
|Random number seed     |set_seed = id     |

# ModelZoo Homepage

- 请访问官方网站 [Homepage](https://gitee.com/mindspore/models).
