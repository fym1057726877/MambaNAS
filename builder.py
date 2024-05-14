import numpy as np
import yaml
import torch
import logging
from models.local_vim import VisionMamba
from models.supernet.super_vim import Super_VisionMamba


def build_model(model_name, cfg_path, logger=None, pretrained=False,
                pretrained_ckpt='', device='cuda', **kwargs):
    cfg = yaml.safe_load(open(cfg_path))
    if logger is None:
        logger = logging.getLogger("./log.json")
    if model_name.lower() == 'vim':
        cfg = cfg['vim']
        logger.info(f"build model_name:{model_name}, config:{cfg}")
        model = VisionMamba(
            img_size=cfg['img_size'],
            patch_size=cfg['patch_size'],
            depth=cfg['depth'],
            embed_dim=cfg['embed_dim'],
            in_chans=cfg['in_chans'],
            num_classes=cfg['num_classes'],
            directions=cfg['directions'],
            **kwargs
        ).to(device)
    elif model_name.lower() == 'super_vim':
        cfg = cfg['super_vim']
        logger.info(f"build model_name:{model_name} \nconfig:{cfg}")
        model = Super_VisionMamba(
            img_size=cfg['img_size'],
            patch_size=cfg['patch_size'],
            depth=cfg['depth'],
            embed_dim=cfg['embed_dim'],
            in_chans=cfg['in_chans'],
            num_classes=cfg['num_classes'],
            directions=cfg['directions'],
            expand_ratio=cfg['expand_ratio'],
            d_state=cfg['d_state'],
            c_kernel_size=cfg['c_kernel_size'],
            num_head=cfg['num_head'],
            mamba_ratio=cfg['mamba_ratio'],
            drop_rate=cfg['drop_rate'],
            drop_path_rate=cfg['drop_path_rate'],
            **kwargs
        ).to(device)
    else:
        raise RuntimeError(f'Model {model_name} not found.')

    if pretrained and pretrained_ckpt != '':
        logger.info(f'Loading pretrained checkpoint from {pretrained_ckpt}')
        ckpt = torch.load(pretrained_ckpt, map_location=device)
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        elif 'model' in ckpt:
            ckpt = ckpt['model']
        missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
        if len(missing_keys) != 0:
            logger.warning(f'Warning:Missing keys in source state dict: {missing_keys}')
        if len(unexpected_keys) != 0:
            logger.warning(f'Warning:Unexpected keys in source state dict: {unexpected_keys}')
    return model


def build_dataloader(dataset_name="tju_pv600"):
    assert dataset_name in ["tju_pv600", "hkpu_pv500", "vera_pv220"]
    if dataset_name == "tju_pv600":
        from datasets.tju_pv600 import getTJUDataLoader
        bacth_size = 30
        image_size = 128
        train_imgs_per_class = 10
        test_imgs_per_class = 5
        val_imgs_per_class = 5
        train_loader, test_loader, val_loader = getTJUDataLoader(image_size=image_size, batch_size=bacth_size,
                                                                 train_imgs_per_class=train_imgs_per_class,
                                                                 test_imgs_per_class=test_imgs_per_class,
                                                                 val_imgs_per_class=val_imgs_per_class)
    elif dataset_name == "hkpu_pv500":
        from datasets.hkpu_pv500 import getPolyUDataLoader
        bacth_size = 25
        image_size = 128
        train_imgs_per_class = 6
        test_imgs_per_class = 3
        val_imgs_per_class = 3
        train_loader, test_loader, val_loader = getPolyUDataLoader(image_size=image_size, batch_size=bacth_size,
                                                                   train_imgs_per_class=train_imgs_per_class,
                                                                   test_imgs_per_class=test_imgs_per_class,
                                                                   val_imgs_per_class=val_imgs_per_class)
    elif dataset_name == "vera_pv220":
        from datasets.vera_pv220 import getVERADataLoader
        bacth_size = 11
        image_size = 128
        train_imgs_per_class = 5
        test_imgs_per_class = 2
        val_imgs_per_class = 3
        train_loader, test_loader, val_loader = getVERADataLoader(image_size=image_size, batch_size=bacth_size,
                                                                  train_imgs_per_class=train_imgs_per_class,
                                                                  test_imgs_per_class=test_imgs_per_class,
                                                                  val_imgs_per_class=val_imgs_per_class)
    else:
        raise NotImplementedError(f"dataset {dataset_name} is not support")
    return train_loader, test_loader, val_loader


def build_candidates(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))['search_space']
    embed_dims = cfg['embed_dim']
    depths = cfg['depth']
    expand_ratios = cfg['expand_ratio']
    d_states = cfg['d_state']
    kernel_sizes = cfg['kernel_size']
    # directions = cfg['direction']

    candidates = []
    for embed_dim in embed_dims:
        for depth in depths:
            for expand_ratio in expand_ratios:
                for d_state in d_states:
                    for kernel_size in kernel_sizes:
                        candidates.append([embed_dim, depth, expand_ratio, d_state, kernel_size])
    return np.array(candidates)
