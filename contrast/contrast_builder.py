
import yaml
import torch
from contrast.contrast_models import *


def build_model(model_name, cfg_path, logger, pretrained=False,
                pretrained_ckpt='', device='cuda'):
    cfg = yaml.safe_load(open(cfg_path))
    cfg_dict = cfg[model_name]
    if model_name == 'Resnet50':
        model = Resnet50(num_classes=cfg_dict['num_classes']).to(device)
    elif model_name == 'FVRASNet':
        model = FVRASNet(num_classes=cfg_dict['num_classes']).to(device)
    elif model_name == 'FVCNN':
        model = FVCNN(num_classes=cfg_dict['num_classes']).to(device)
    elif model_name == 'LightweightCNN':
        model = LightweightCNN(num_classes=cfg_dict['num_classes']).to(device)
    elif model_name == 'MSMDGANetCNN':
        model = MSMDGANetCNN(num_classes=cfg_dict['num_classes']).to(device)
    else:
        raise RuntimeError(f'Model {model_name} not found.')
    logger.info(f"build model_name:{model_name} \nconfig:{cfg_dict}")

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


def build_dataloader(dataset="PV600"):
    if dataset == "PV600":
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
    elif dataset == "PV500":
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
    else:
        raise NotImplementedError(f"dataset {dataset} is not support")
    return train_loader, test_loader, val_loader



