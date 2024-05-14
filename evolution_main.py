

import yaml
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import random
import argparse
from os.path import join
from builder import build_dataloader, build_model
from models.evolution import EvolutionSearcher
from utils import *


def get_args_parser():
    parser = argparse.ArgumentParser('evolution process', add_help=False)
    parser.add_argument('--device', default="cuda:0", type=str)
    parser.add_argument('--dataset-name', default="tju_pv600", type=str)
    parser.add_argument('--seed', default=42, type=int)

    # evolution search parameters
    parser.add_argument('--max-epochs', type=int, default=20)
    parser.add_argument('--select-num', type=int, default=10)
    parser.add_argument('--population-num', type=int, default=50)
    parser.add_argument('--m_prob', type=float, default=0.2)
    parser.add_argument('--s_prob', type=float, default=0.4)
    parser.add_argument('--crossover-num', type=int, default=25)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--mutation-num', type=int, default=25)
    parser.add_argument('--param-limits', type=float, default=15)
    parser.add_argument('--min-param-limits', type=float, default=0)

    # Model parameters
    parser.add_argument('--model-name', default='super_vim', type=str, help='Name of model to search')

    return parser.parse_args()


def main(args):
    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cudnn.benchmark = True

    log_path = join(project_path, "logs", f"evolution.log.json")
    logger = get_logger(file_name=log_path)
    logger.info("\n\n")
    logger.info(args)
    train_loader, test_loader, val_loader = build_dataloader(dataset_name=args.dataset_name)

    (evolution_output_dir, evolution_checkpoint_path,
     model_cfg_path, model_save_path) = get_default_path(args.dataset_name)
    model_save_path = join(model_save_path,  f'{args.model_name}.pth')

    model = build_model(model_name=args.model_name, cfg_path=model_cfg_path, pretrained=True,
                        pretrained_ckpt=model_save_path, logger=logger, device=device)

    search_space = yaml.safe_load(open(model_cfg_path))['search_space']

    choices = {'embed_dim': search_space['embed_dim'], 'expand_ratio': search_space['expand_ratio'],
               'depth': search_space['depth'], 'd_state': search_space['d_state'],
               'c_kernel_size': search_space['c_kernel_size'], 'mamba_ratio': search_space['mamba_ratio'],
               'num_head': search_space['num_head']}
    logger.info(f"search_space:{choices}")

    t = time.time()
    searcher = EvolutionSearcher(args=args, device=device, model=model, choices=choices, logger=logger,
                                 val_loader=val_loader, test_loader=test_loader, output_dir=evolution_output_dir)

    searcher.search()

    print(f'total searching time = {time.time() - t / 3600:.2f} hours')


if __name__ == '__main__':
    args = get_args_parser()
    main(args)
