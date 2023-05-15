# Copyright 2021-2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
""" train PSPNet and get checkpoint files """
import os
import ast
import argparse
import json
import src.utils.functions_args as fa
from src.model import pspnet
from src.model.cell import Aux_CELoss_Cell
from src.dataset import pt_dataset
from src.dataset import pt_transform as transform
from src.utils.lr import poly_lr
from src.utils.get_misc import set_device, get_rank_info
from src.utils.metric_and_evalcallback import pspnet_metric
from src.utils.eval_callback import EvalCallBack, SegEvalCallback
from moxing_adapter import sync_data, sync_multi_data
import mindspore
from mindspore import nn
from mindspore import context
from mindspore import Tensor
from mindspore.common import set_seed
from mindspore.train.model import Model
from mindspore.train.callback import TimeMonitor
from mindspore.train.loss_scale_manager import FixedLossScaleManager
import mindspore.dataset as ds


def get_parser():
    """
    Read parameter file
        -> for ADE20k: ./config/ade20k_pspnet50.yaml
        -> for voc2012: ./config/voc2012_pspnet50.yaml
    """
    parser = argparse.ArgumentParser(description='MindSpore Semantic Segmentation')
    parser.add_argument('--config', type=str, default="src/config/voc2012_pspnet50.yaml",
                        help='config file')
    parser.add_argument('--model_art', type=ast.literal_eval, default=True,
                        help='train on modelArts or not, default: True')

    parser.add_argument('--data_url', default='',
                        help='path to training/inference dataset folder')
    parser.add_argument('--multi_data_url', default='',
                        help='path to multi dataset')
    parser.add_argument('--train_url', default='',
                        help='output folder to save/load')
    parser.add_argument('--ckpt_url', default='',
                        help='Location of training outputs.')
    parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'CPU'],
                        help='device where the code will be implemented (default: Ascend),'
                             'if to use the CPU on the Qizhi platform:device_target=CPU')
    args_ = parser.parse_args()

    print("multi_data_url:->> ", args_.multi_data_url)
    args_.config = os.path.join(os.path.dirname(__file__), args_.config)
    assert args_.config is not None
    cfg = fa.load_cfg_from_cfg_file(args_.config)

    Model_Art = False
    if args_.model_art:
        Model_Art = True
        cfg.multi_data_url = args_.multi_data_url
        print("Start load data!")
        if args_.multi_data_url:
            ckpt_url = json.loads(args_.ckpt_url)[0]["model_url"]
            sync_multi_data(args_.multi_data_url, cfg.art_root_path)
            sync_data(ckpt_url, cfg.art_pretrain_path)
        else:
            cfg.train_url = args_.train_url
            sync_data(args_.data_url, cfg.art_root_path)
            sync_data(args_.ckpt_url, cfg.art_pretrain_path)

    return cfg, Model_Art


def create_dataset(args, purpose, data_root, data_list, batch_size=8):
    """ get dataset """
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    num_parallel_workers = 4
    device_num, rank_id = get_rank_info()

    if purpose == 'train':
        cur_transform = transform.Compose([
            transform.RandScale([0.5, 2.0]),
            transform.RandRotate([-10, 10], padding=mean, ignore_label=255),
            transform.RandomGaussianBlur(),
            transform.RandomHorizontalFlip(),
            transform.Crop([args.train_h, args.train_w], crop_type='rand', padding=mean, ignore_label=255),
            transform.Normalize(mean=mean, std=std, is_train=True)])
        data = pt_dataset.SemData(
            split=purpose, data_root=data_root,
            data_list=data_list,
            transform=cur_transform,
            data_name=args.data_name
        )
        dataset = ds.GeneratorDataset(data, column_names=["data", "label"], num_parallel_workers=num_parallel_workers,
                                      shuffle=True, num_shards=device_num, shard_id=rank_id)
        dataset = dataset.batch(batch_size, drop_remainder=True, num_parallel_workers=num_parallel_workers)
    else:
        cur_transform = transform.Compose([
            transform.Crop([args.train_h, args.train_w], crop_type='center', padding=mean, ignore_label=255),
            transform.Normalize(mean=mean, std=std, is_train=True)])
        data = pt_dataset.SemData(
            split=purpose, data_root=data_root,
            data_list=data_list,
            transform=cur_transform,
            data_name=args.data_name
        )

        dataset = ds.GeneratorDataset(data, column_names=["data", "label"], num_parallel_workers=num_parallel_workers,
                                      shuffle=False, num_shards=device_num, shard_id=rank_id)
        dataset = dataset.batch(batch_size, drop_remainder=False, num_parallel_workers=num_parallel_workers)

    return dataset


def psp_train():
    """ Train process """
    args, Model_Art = get_parser()

    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    context.set_context(enable_graph_kernel=False)
    rank = set_device(args)
    set_seed(rank)

    if Model_Art:
        pre_path = args.art_pretrain_path
        if args.multi_data_url:
            data_path = args.art_data_root_zs
            train_list_path = args.art_train_list_zs
            val_list_path = args.art_val_list_zs
        else:
            data_path = args.art_data_root
            train_list_path = args.art_train_list
            val_list_path = args.art_val_list
    else:
        pre_path = args.pretrain_path
        data_path = args.data_root
        train_list_path = args.train_list
        val_list_path = args.val_list

    PSPNet = pspnet.PSPNet(num_classes=args.classes, backbone=args.backbone, pretrained=True,
                           pretrained_path=pre_path, deep_base=True)

    train_dataset = create_dataset(args, 'train', data_path, train_list_path, batch_size=args.batch_size)
    validation_dataset = create_dataset(args, 'val', data_path, val_list_path, batch_size=args.batch_size_val)

    # loss
    train_net_loss = Aux_CELoss_Cell(args.classes, ignore_label=255)

    steps_per_epoch = train_dataset.get_dataset_size()  # Return the number of batches in an epoch.
    total_train_steps = steps_per_epoch * args.epochs

    lr_iter = poly_lr(args.base_lr, total_train_steps, total_train_steps, end_lr=0.0, power=0.9)
    lr_iter_ten = poly_lr(args.base_lr * 10, total_train_steps, total_train_steps, end_lr=0.0, power=0.9)

    modules_ori = [PSPNet.layer0, PSPNet.layer1, PSPNet.layer2, PSPNet.layer3, PSPNet.layer4]
    modules_new = [PSPNet.ppm, PSPNet.cls, PSPNet.aux]
    group_params = []
    for module in modules_ori:
        group_params.append(dict(params=module.trainable_params(), lr=Tensor(lr_iter, mindspore.float32)))
    for module in modules_new:
        group_params.append(dict(params=module.trainable_params(), lr=Tensor(lr_iter_ten, mindspore.float32)))

    opt = nn.SGD(params=group_params, momentum=args.momentum, weight_decay=args.weight_decay, loss_scale=1024)

    # loss scale
    manager_loss_scale = FixedLossScaleManager(1024, False)
    m_metric = {'val_loss': pspnet_metric(args.classes, 255)}
    model = Model(PSPNet, train_net_loss, optimizer=opt, loss_scale_manager=manager_loss_scale, metrics=m_metric)

    time_cb = TimeMonitor(data_size=steps_per_epoch)
    epoch_per_eval = {"epoch": [], "val_loss": []}
    eval_cb = EvalCallBack(model, validation_dataset, 1, epoch_per_eval)
    output_url = '/cache/output/'
    eval_iou = SegEvalCallback(validation_dataset, PSPNet, args.classes, start_epoch=0,
                               save_path=output_url, interval=1, rank=rank)

    model.train(args.epochs, train_dataset, callbacks=[time_cb, eval_cb, eval_iou], dataset_sink_mode=True)

    if Model_Art and not args.multi_data_url:
        print("######### upload to OBS #########")
        sync_data(output_url, args.train_url)


if __name__ == "__main__":
    psp_train()
