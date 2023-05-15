import torch

# pretrained_weight = torch.load('resnet50_v2.pth', map_location=lambda storage, loc: storage)
# out_put = []
# for name, param in pretrained_weight.items():
#     if 'num_batches_tracked' in name:
#         continue
#     print(name, param.shape)
#     out_put.append(name)
# out_put = '\n'.join(out_put)
# with open("key_torch_resnet_50.txt", 'w') as f:
#     f.write(out_put)

pretrained_weight = torch.load('train_epoch_50.pth', map_location=lambda storage, loc: storage)["state_dict"]
out_put = []
for name, param in pretrained_weight.items():
    if 'num_batches_tracked' in name:
        continue
    if "module" in name:
        name = name[7:]
    print(name, param.shape)
    out_put.append(name)
out_put = '\n'.join(out_put)
with open("key_torch_train_epoch_50.txt", 'w') as f:
    f.write(out_put)
