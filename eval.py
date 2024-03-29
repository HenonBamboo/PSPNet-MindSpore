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
""" VOC2012 DATASET EVALUATE """
import os
import ast
import time
import logging
import argparse
import cv2
import numpy
import json
from src.dataset import pt_dataset, pt_transform
import src.utils.functions_args as fa
from src.utils.p_util import AverageMeter, intersectionAndUnion, check_makedirs, colorize
from moxing_adapter import sync_data, sync_multi_data
import mindspore.numpy as np
from mindspore import Tensor
import mindspore.dataset as ds
from mindspore import context
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.train.serialization import load_param_into_net, load_checkpoint

cv2.ocl.setUseOpenCL(False)
device_id = int(os.getenv('DEVICE_ID', '0'))
Model_Art = False


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

    parser.add_argument('--data_url', type=str, default='',
                        help='path to training/inference dataset folder')
    parser.add_argument('--multi_data_url', type=str, default='',
                        help='path to multi dataset')
    parser.add_argument('--train_url', type=str, default='',
                        help='output folder to save/load')
    parser.add_argument('--ckpt_url', type=str, default='',
                        help='Location of training outputs.')
    parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'CPU'],
                        help='device where the code will be implemented (default: Ascend),'
                             'if to use the CPU on the Qizhi platform:device_target=CPU')
    args_ = parser.parse_args()

    print("multi_data_url:->> ", args_.multi_data_url)
    args_.config = os.path.join(os.path.dirname(__file__), args_.config)
    assert args_.config is not None
    cfg = fa.load_cfg_from_cfg_file(args_.config)
    global Model_Art
    if args_.model_art:
        Model_Art = True
        cfg.train_url = args_.train_url
        cfg.multi_data_url = args_.multi_data_url
        print("Start load data!")
        if args_.multi_data_url:
            ckpt_url = json.loads(args_.ckpt_url)[0]["model_url"]
            sync_multi_data(args_.multi_data_url, cfg.art_root_path)
            sync_data(ckpt_url, cfg.art_ckpt)
        else:
            sync_data(args_.data_url, cfg.art_root_path)
            sync_data(args_.ckpt_url, cfg.art_ckpt)
    return cfg


def get_logger():
    """ logger """
    logger_name = "main-logger"
    logger_ = logging.getLogger(logger_name)
    logger_.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger_.addHandler(handler)
    return logger_


def check(local_args):
    """ check args """
    assert local_args.classes > 1
    assert local_args.split in ['train', 'val', 'test']
    if local_args.arch == 'psp':
        assert (local_args.train_h - 1) % 8 == 0 and (local_args.train_w - 1) % 8 == 0
    else:
        raise Exception('architecture not supported {} yet'.format(local_args.arch))


def main():
    """ The main function of the evaluate process """
    check(args)
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=device_id,
                        save_graphs=False)
    logger.info("=> creating model ...")
    logger.info("Classes: %s", args.classes)

    if Model_Art:
        pre_path = args.art_pretrain_path
        ckpt = args.art_ckpt
        if args.multi_data_url:
            data_path = args.art_data_root_zs
            val_list_path = args.art_val_list_zs
            color_txt = args.art_color_txt_zs
            name_txt = args.art_name_txt_zs
        else:
            data_path = args.art_data_root
            val_list_path = args.art_val_list
            color_txt = args.art_color_txt
            name_txt = args.art_name_txt
    else:
        pre_path = args.pretrain_path
        data_path = args.data_root
        val_list_path = args.val_list
        color_txt = args.color_txt
        name_txt = args.name_txt
        ckpt = args.ckpt

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    gray_folder = os.path.join(args.result_path, 'gray')
    color_folder = os.path.join(args.result_path, 'color')

    test_transform = pt_transform.Compose([pt_transform.Normalize(mean=mean, std=std, is_train=False)])
    test_data = pt_dataset.SemData(
        split='val', data_root=data_path,
        data_list=val_list_path,
        transform=test_transform,
        data_name=args.data_name)
    test_loader = ds.GeneratorDataset(test_data, column_names=["data", "label"], shuffle=False)
    test_loader.batch(1)

    colors = numpy.loadtxt(color_txt).astype('uint8')
    names = [line.rstrip('\n') for line in open(name_txt)]

    from src.model import pspnet
    PSPNet = pspnet.PSPNet(num_classes=args.classes, backbone=args.backbone, pretrained=False,
                           pretrained_path=pre_path, deep_base=True)

    ms_checkpoint = load_checkpoint(ckpt)
    load_param_into_net(PSPNet, ms_checkpoint, strict_load=True)

    PSPNet.set_train(False)
    val(test_loader, test_data.data_list, PSPNet, args.classes, mean, std, args.base_size, args.test_h,
        args.test_w, args.scales, gray_folder, color_folder, colors)
    cal_acc(test_data.data_list, gray_folder, args.classes, names)


def net_process(model, image, mean, std=None, flip=True):
    """ Give the input to the model"""
    transpose = ops.Transpose()
    input_ = transpose(image, (2, 0, 1))  # (473, 473, 3) -> (3, 473, 473)
    mean = np.array(mean)
    std = np.array(std)
    if std is None:
        input_ = input_ - mean[:, None, None]
    else:
        input_ = (input_ - mean[:, None, None]) / std[:, None, None]

    expand_dim = ops.ExpandDims()
    input_ = expand_dim(input_, 0)
    if flip:
        if args.device_target.upper() == 'CPU':
            flip_input = np.flip(input_, axis=3)
        else:
            flip_ = ops.ReverseV2(axis=[3])
            flip_input = flip_(input_)
        concat = ops.Concat(axis=0)
        input_ = concat((input_, flip_input))

    model.set_train(False)
    output = model(input_)
    _, _, h_i, w_i = input_.shape
    _, _, h_o, w_o = output.shape
    if (h_o != h_i) or (w_o != w_i):
        bi_linear = nn.ResizeBilinear()
        output = bi_linear(output, size=(h_i, w_i), align_corners=True)
    softmax = nn.Softmax(axis=1)
    output = softmax(output)
    if flip:
        if args.device_target.upper() == 'CPU':
            output = (output[0] + np.flip(output[1], axis=2)) / 2
        else:
            flip_ = ops.ReverseV2(axis=[2])
            output = (output[0] + flip_(output[1])) / 2
    else:
        output = output[0]
    output = transpose(output, (1, 2, 0))  # Tensor
    output = output.asnumpy()
    return output


def scale_process(model, image, classes, crop_h, crop_w, h, w, mean, std=None, stride_rate=2 / 3):
    """ Process input size """
    ori_h, ori_w, _ = image.shape
    pad_h = max(crop_h - ori_h, 0)
    pad_w = max(crop_w - ori_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half,
                                   cv2.BORDER_CONSTANT, value=mean)

    new_h, new_w, _ = image.shape
    image = Tensor.from_numpy(image)
    stride_h = int(numpy.ceil(crop_h * stride_rate))
    stride_w = int(numpy.ceil(crop_w * stride_rate))
    grid_h = int(numpy.ceil(float(new_h - crop_h) / stride_h) + 1)
    grid_w = int(numpy.ceil(float(new_w - crop_w) / stride_w) + 1)
    prediction_crop = numpy.zeros((new_h, new_w, classes), dtype=float)
    count_crop = numpy.zeros((new_h, new_w), dtype=float)
    for index_h in range(0, grid_h):
        for index_w in range(0, grid_w):
            s_h = index_h * stride_h
            e_h = min(s_h + crop_h, new_h)
            s_h = e_h - crop_h
            s_w = index_w * stride_w
            e_w = min(s_w + crop_w, new_w)
            s_w = e_w - crop_w
            image_crop = image[s_h:e_h, s_w:e_w].copy()
            count_crop[s_h:e_h, s_w:e_w] += 1
            prediction_crop[s_h:e_h, s_w:e_w, :] += net_process(model, image_crop, mean, std)
    prediction_crop /= numpy.expand_dims(count_crop, 2)
    prediction_crop = prediction_crop[pad_h_half:pad_h_half + ori_h, pad_w_half:pad_w_half + ori_w]
    prediction = cv2.resize(prediction_crop, (w, h), interpolation=cv2.INTER_LINEAR)
    return prediction


def val(test_loader, data_list, model, classes, mean, std, base_size, crop_h, crop_w, scales, gray_folder,
        color_folder, colors):
    """ Generate evaluate image """
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    data_time = AverageMeter()
    batch_time = AverageMeter()
    model.set_train(False)
    end = time.time()
    for i, (input_, _) in enumerate(test_loader):
        data_time.update(time.time() - end)
        input_ = input_.asnumpy()
        image = numpy.transpose(input_, (1, 2, 0))
        h, w, _ = image.shape
        prediction = numpy.zeros((h, w, classes), dtype=float)
        for scale in scales:
            long_size = round(scale * base_size)
            new_h = long_size
            new_w = long_size
            if h > w:
                new_w = round(long_size / float(h) * w)
            else:
                new_h = round(long_size / float(w) * h)

            image_scale = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            prediction += scale_process(model, image_scale, classes, crop_h, crop_w, h, w, mean, std)
        prediction /= len(scales)
        prediction = numpy.argmax(prediction, axis=2)
        batch_time.update(time.time() - end)
        end = time.time()
        if ((i + 1) % 10 == 0) or (i + 1 == len(data_list)):
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}).'.format(i + 1, len(data_list),
                                                                                    data_time=data_time,
                                                                                    batch_time=batch_time))
        check_makedirs(gray_folder)
        check_makedirs(color_folder)
        gray = numpy.uint8(prediction)
        color = colorize(gray, colors)
        image_path, _ = data_list[i]
        image_path = image_path.replace('\\', '/')
        image_name = image_path.split('/')[-1].split('.')[0]
        gray_path = os.path.join(gray_folder, image_name + '.png')
        color_path = os.path.join(color_folder, image_name + '.png')
        cv2.imwrite(gray_path, gray)
        color.save(color_path)
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')


def cal_acc(data_list, pred_folder, classes, names):
    """ Calculation evaluating indicator """
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    for i, (image_path, target_path) in enumerate(data_list):
        # image_name = image_path.split('/')[-1].split('.')[0]
        image_name = image_path.split('/')[-1][:-4]
        pred = cv2.imread(os.path.join(pred_folder, image_name + '.png'), cv2.IMREAD_GRAYSCALE)
        target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
        if args.prefix == 'ADE':
            target -= 1
        intersection, union, target = intersectionAndUnion(pred, target, classes)
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)
        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        logger.info(
            'Evaluating {0}/{1} on image {2}, accuracy {3:.4f}.'.format(i + 1, len(data_list), image_name + '.png',
                                                                        accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = numpy.mean(iou_class)
    mAcc = numpy.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    logger.info('Eval result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    for i in range(classes):
        logger.info('Class_{} result: iou/accuracy {:.4f}/{:.4f}, name: {}.'.format(i, iou_class[i], accuracy_class[i],
                                                                                    names[i]))


if __name__ == '__main__':
    args = get_parser()
    logger = get_logger()
    main()
