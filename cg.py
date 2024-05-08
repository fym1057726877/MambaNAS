import argparse
import os
import random
from os.path import join

import numpy as np
import torch
import logging
from logging import handlers

from sklearn.metrics import roc_curve
from timm.models.layers import DropPath, PatchEmbed
from models.mamba_local.mamba.multi_mamba import MultiMamba
from torch.nn import functional as F
from torch import nn
from models.supernet.super_ops import Super_Conv1d
import yaml
from utils import get_logger, project_path

# xz: torch.Size([60, 64, 320])
# A: torch.Size([960, 64])
# D: torch.Size([960])
# conv1d.weight: torch.Size([960, 1, 4])
# conv1d.bias: torch.Size([960])
# x_proj.weight: torch.Size([960, 1, 4])
# dt_proj.weight: torch.Size([960, 1, 4])
# dt_proj.bias: torch.Size([960])

# A: torch.Size([384, 16])
# D: torch.Size([384])
# conv1d.weight: torch.Size([384, 1, 4])
# conv1d.bias: torch.Size([384])
# x_proj.weight: torch.Size([44, 384])
# dt_proj.weight: torch.Size([384, 12])
# dt_proj.bias: torch.Size([384])

# from models.local_vim import VisionMamba
#
# model = VisionMamba(img_size=128, in_chans=1, patch_size=16).to("cuda")
#
# x = torch.rand(4, 1, 128, 128).to("cuda")
#
# print(model(x).shape)

# from models.supernet.super_vim import Super_VisionMamba
#
# device = "cuda"
# model = Super_VisionMamba(
#             img_size=128,
#             patch_size=32,
#             depth=4,
#             embed_dim=320,
#             in_chans=1,
#             num_classes=600,
#             directions=None,
#             expand_ratio=3,
#             d_state=64,
#             kernel_size=4,
# ).to(device)
# total_paras, info = model.count_parameters()
# print(info)
# model.count_parameters_and_flops(input_shape=(2, 1, 128, 128))

# paras = model.get_sampled_params_numel()
# flops = model.get_complexity(sequence_length=64)
# print(f"{paras/1e6:.2f}M")
# print(f"{flops/1e9:.2f}G")

# y_true = [2, 0, 1, 0, 1]
# y_score = [[0.2, 0.3, 0.5], [0.2, 0.5, 0.3], [0.4, 0.5, 0.1], [0.3, 0.5, 0.2], [0.1, 0.8, 0.1]]
# fpr, tpr, thresholds = roc_curve(y_true, y_score)
#
# print("fpr:\n", fpr)
# print("tpr:\n", tpr)
# print("thresholds:\n", thresholds)


# from measure import get_score
# from builder import build_dataloader, build_model
#
# train_loader, test_loader, val_loader = build_dataloader(dataset='PV600')
# model = build_model(model_name='vim',
#                     cfg_path='E:/fym/code/Pythonproject/MambaNAS/models/configs/vim/tju600.yaml',
#                     pretrained=True,
#                     pretrained_ckpt='E:/fym/code/Pythonproject/MambaNAS/ckpts/vim.pth'
#                     )
# y_true, y_pred = get_score(model, test_loader)
# print(y_true.shape)
# print(y_pred.shape)


# a = np.array([[0.2, 0.5, 0.3], [0.1, 0.8, 0.1], [0.5, 0.4, 0.1]], dtype=np.float32)
# l = np.array([1, 0, 2], dtype=np.int16)
#
# print([a[i][l[i]] for i in range(len(l))])
#
# b = [a[j][i] for j in range(len(a)) for i in range(len(a)) if i != l[j]]
# print(b)

# model_cfg_path = join(project_path, 'models', 'configs', 'vim', 'tju600.yaml')
# cfg = yaml.safe_load(open(model_cfg_path))['search_space']
#
# choices = {'embed_dim': cfg['embed_dim'], 'expand_ratio': cfg['expand_ratio'],
#            'depth': cfg['depth'], 'd_state': cfg['d_state'], 'kernel_size': cfg['kernel_size']}
# cand_tuple = list()
# dimensions = ['expand_ratio', 'd_state', 'kernel_size']
# depth = random.choice(choices['depth'])
# cand_tuple.append(depth)
# for dimension in dimensions:
#     for i in range(depth):
#         cand_tuple.append(random.choice(choices[dimension]))
# cand_tuple.append(random.choice(choices['embed_dim']))
# print(tuple(cand_tuple))
