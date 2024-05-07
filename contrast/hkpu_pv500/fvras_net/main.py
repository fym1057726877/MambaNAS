
import torch
import argparse
from os.path import join
from timm.data import Mixup
from torch.nn import CrossEntropyLoss
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from utils import get_logger, get_project_path
from engine import train_multi_epochs
from builder import build_model, build_dataloader


def get_args_parser():
    parser = argparse.ArgumentParser("training and evaluation script")
    parser.add_argument('--num-classes', default=600, type=int)
    parser.add_argument('--total-epochs', default=500, type=int)
    parser.add_argument('--device', default="cuda:0", type=str)
    parser.add_argument('--dataset', default="PV600", type=str)

    # Model parameters
    parser.add_argument('--model-name', default='super_vim', type=str, help='Name of model to train')
    parser.add_argument('--model-cfg',
                        default='E:/fym/code/Pythonproject/MambaNAS/models/configs/vim/tju600.yaml',
                        type=str, help='model configs file')
    parser.add_argument('--model-save-path',
                        default='E:/fym/code/Pythonproject/MambaNAS/ckpts/super_vim.pth',
                        type=str, help='model_save_path')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=1.0,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')

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
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
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

    # multi gpu train
    parser.add_argument('--ddp', type=bool, default=False, help='multi gpu train')

    return parser.parse_args()


def main(args):
    device = args.device
    log_path = join(get_project_path("MambaNAS"), "logs", f"{args.model_name}.log.json")
    logger = get_logger(file_name=log_path)
    logger.info("\n\n")
    logger.info(args)
    train_loader, test_loader, val_loader = build_dataloader(dataset=args.dataset)
    model = build_model(model_name=args.model_name, cfg_path=args.model_cfg,
                        logger=logger, device=device)

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)

    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = CrossEntropyLoss()

    optimizer = create_optimizer(args, model)
    lr_scheduler, _ = create_scheduler(args, optimizer)

    if args.ddp:
        model = torch.nn.parallel.DataParallel(model)

    train_multi_epochs(
        model=model,
        model_save_path=args.model_save_path,
        train_loader=train_loader,
        test_loader=test_loader,
        val_loader=val_loader,
        mixup_fn=mixup_fn,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        total_epochs=args.total_epochs,
        device=device,
        logger=logger,
        train_progress=False,
        test_progress=False,
        val_progress=False,
    )


if __name__ == '__main__':
    args = get_args_parser()
    main(args)
