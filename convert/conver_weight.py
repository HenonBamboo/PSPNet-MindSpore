"""weight convert"""
import os
import torch
from mindspore import Tensor, save_checkpoint, dtype, Parameter


def get_torch_key():
    with open("key_torch_train_epoch_50.txt", 'r') as f:
        keys_mindspore = f.readlines()
    keys_mindspore = [key.split("\n")[0] for key in keys_mindspore]
    print(f"params num: {len(keys_mindspore)}")
    keys_mindspore_top = []
    keys_mindspore_middle = []
    keys_mindspore_tail = []
    for key in keys_mindspore:
        if "transitions" in key:
            keys_mindspore_tail.append(key)
        elif "running_mean" in key or "running_var" in key:
            keys_mindspore_middle.append(key)
        else:
            keys_mindspore_top.append(key)
    keys_mindspore_all = keys_mindspore_top + keys_mindspore_middle + keys_mindspore_tail
    return keys_mindspore_all


def main():
    keys_torch = get_torch_key()

    pretrained_weight = torch.load('train_epoch_50.pth', map_location=lambda storage, loc: storage)["state_dict"]

    params_list = []
    for key_torch in keys_torch:
        params_dict = {}
        weight = pretrained_weight["module." + key_torch].numpy()

        if "bn" in key_torch or "cls.1" in key_torch or "downsample.1" in key_torch or "aux.1" in key_torch \
                or "ppm.features.0.2" in key_torch or "ppm.features.1.2" in key_torch\
                or "ppm.features.2.2" in key_torch or "ppm.features.3.2" in key_torch\
                or "layer0.1" in key_torch or "layer0.4" in key_torch or "layer0.7" in key_torch:
            if "weight" in key_torch:
                key_torch = key_torch.replace(".weight", ".gamma")
            elif "bias" in key_torch:
                key_torch = key_torch.replace(".bias", ".beta")
            elif "mean" in key_torch:
                key_torch = key_torch.replace(".running_mean", ".moving_mean")
            elif "var" in key_torch:
                key_torch = key_torch.replace(".running_var", ".moving_variance")

        print(key_torch)
        params_dict["data"] = Parameter(Tensor(weight, dtype=dtype.float32), requires_grad=True)
        params_dict["name"] = key_torch
        params_list.append(params_dict)

    path = "weight_train_epoch_50.ckpt"
    if os.path.exists(path):
        os.remove(path)
    save_checkpoint(params_list, path)


if __name__ == '__main__':
    main()
