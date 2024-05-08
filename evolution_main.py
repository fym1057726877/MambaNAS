

import yaml
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import random
import argparse
from os.path import join
from engine import evaluate
from builder import build_dataloader, build_model
from models.evolution import EvolutionSearcher
from utils import project_path, get_logger


def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--num-classes', default=600, type=int)
    parser.add_argument('--total-epochs', default=500, type=int)
    parser.add_argument('--device', default="cuda:0", type=str)
    parser.add_argument('--dataset', default="PV600", type=str)
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
    parser.add_argument('--param-limits', type=float, default=23)
    parser.add_argument('--min-param-limits', type=float, default=18)

    # Model parameters
    parser.add_argument('--model-name', default='vim', type=str, help='Name of model to train')
    parser.add_argument('--model-cfg',
                        default='/home/yons/桌面/fym/MambaNAS/models/configs/vim/tju600.yaml',
                        type=str, help='model configs file')
    parser.add_argument('--model-save-path',
                        default='/home/yons/桌面/fym/MambaNAS/ckpts/vim.pth',
                        type=str, help='model_save_path')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--lr-power', type=float, default=1.0,
                        help='power of the polynomial lr scheduler')
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=1.0,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=0.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--no-amp', action='store_false', dest='amp')
    parser.set_defaults(amp=True)

    return parser.parse_args()


def main(args):
    print(args)
    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cudnn.benchmark = True

    args.prefetcher = not args.no_prefetcher

    log_path = join(project_path, "logs", f"{args.model_name}.log.json")
    logger = get_logger(file_name=log_path)
    logger.info("\n\n")
    logger.info(args)
    train_loader, test_loader, val_loader = build_dataloader(dataset=args.dataset)

    logger.info(f"Creating {args.model_name}")
    model_save_path = args.model_save_path + args.model_name + '.pth'
    model = build_model(model_name=args.model_name, cfg_path=args.model_cfg, pretrained=True,
                        pretrained_ckpt=model_save_path, logger=logger, device=device)

    cfg = yaml.safe_load(open(args.model_cfg))['search_space']
    logger.info(cfg)

    choices = {'embed_dim': cfg['embed_dim'], 'expand_ratio': cfg['expand_ratio'],
               'depth': cfg['depth'], 'd_state': cfg['d_state'], 'kernel_size': cfg['kernel_size']}

    t = time.time()
    searcher = EvolutionSearcher(args=args, device=device, model=model, choices=choices,
                                 val_loader=val_loader, test_loader=test_loader, output_dir=args.output_dir)

    searcher.search()

    print('total searching time = {:.2f} hours'.format((time.time() - t) / 3600))


if __name__ == '__main__':
    args = get_args_parser()
    main(args)
