# Copyright 2021 Huawei Technologies Co., Ltd
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
"""
Functions for input transform
"""
import random
import numbers
import collections
import numpy as np
import cv2


class Compose:
    """ compose the process functions """

    def __init__(self, segtransform):
        self.segtransform = segtransform

    def __call__(self, image):
        for t in self.segtransform:
            image = t(image)
        return image


class Normalize:
    """
    Normalize tensor with mean and standard deviation along channel:

        channel = (channel - mean) / std

    """

    def __init__(self, mean, std=None, is_train=True):
        if std is None:
            assert mean
        else:
            assert len(mean) == len(std)
        self.mean = np.array(mean)
        self.std = np.array(std)
        self.is_train = is_train

    def __call__(self, image):
        if not isinstance(image, np.ndarray):
            raise RuntimeError("segtransform.ToTensor() only handle np.ndarray"
                               "[eg: data read by cv2.imread()].\n")
        if len(image.shape) > 3 or len(image.shape) < 2:
            raise RuntimeError("segtransform.ToTensor() only handle np.ndarray with 3 dims or 2 dims.\n")
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        image = np.transpose(image, (2, 0, 1))  # (473, 473, 3) -> (3, 473, 473)

        if self.is_train:
            if self.std is None:
                image = image - self.mean[:, None, None]
            else:
                image = (image - self.mean[:, None, None]) / self.std[:, None, None]
        return image


class Crop:
    """Crops the given ndarray image (H*W*C or H*W).
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
        int instead of sequence like (h, w), a square crop (size, size) is made.
    """

    def __init__(self, size, crop_type='center', padding=None, ignore_label=255):
        # [473, 473], 'rand', padding=mean, ignore255
        if isinstance(size, int):
            self.crop_h = size
            self.crop_w = size
        elif isinstance(size, collections.Iterable) and len(size) == 2 \
                and isinstance(size[0], int) and isinstance(size[1], int) \
                and size[0] > 0 and size[1] > 0:
            self.crop_h = size[0]
            self.crop_w = size[1]
        else:
            raise RuntimeError("crop size error.\n")
        if crop_type in ('center', 'rand'):
            self.crop_type = crop_type
        else:
            raise RuntimeError("crop type error: rand | center\n")
        if padding is None:
            self.padding = padding
        elif isinstance(padding, list):
            if all(isinstance(i, numbers.Number) for i in padding):
                self.padding = padding
            else:
                raise RuntimeError("padding in Crop() should be a number list\n")
            if len(padding) != 3:
                raise RuntimeError("padding channel is not equal with 3\n")
        else:
            raise RuntimeError("padding in Crop() should be a number list\n")
        if isinstance(ignore_label, int):
            self.ignore_label = ignore_label
        else:
            raise RuntimeError("ignore_label should be an integer number\n")

    def __call__(self, image):
        h, w, c = image.shape
        pad_h = max(self.crop_h - h, 0)
        pad_w = max(self.crop_w - w, 0)
        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)
        if pad_h > 0 or pad_w > 0:
            if self.padding is None:
                raise RuntimeError("segtransform.Crop() need padding while padding argument is None\n")
            image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half,
                                       cv2.BORDER_CONSTANT, value=self.padding)

        h, w, c = image.shape
        if self.crop_type == 'rand':
            h_off = random.randint(0, h - self.crop_h)
            w_off = random.randint(0, w - self.crop_w)
        else:
            h_off = int((h - self.crop_h) / 2)
            w_off = int((w - self.crop_w) / 2)
        image = image[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w]
        return image


class RGB2BGR:
    """
    Converts image from RGB order to BGR order
    """

    def __init__(self):
        pass

    def __call__(self, image, label):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, label


class BGR2RGB:
    """
    Converts image from BGR order to RGB order
    """
    def __init__(self):
        pass

    def __call__(self, image, label):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, label
