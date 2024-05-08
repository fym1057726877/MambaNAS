
import torch
from logging import Logger
import sys
import math
import random
from typing import Optional
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from time import time
from torch import nn, optim
from timm.data import Mixup
from timm.scheduler import scheduler
from matplotlib import pyplot as plt


def sample_configs(choices):
    config = {}
    dimensions = ['d_state', 'expand_ratio']
    depth = random.choice(choices['depth'])
    for dimension in dimensions:
        config[dimension] = [random.choice(choices[dimension]) for _ in range(depth)]

    config['embed_dim'] = [random.choice(choices['embed_dim'])] * depth

    config['depth'] = depth
    return config


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    target = target.reshape(1, -1).expand_as(pred)
    correct = pred.eq(target).detach().cpu()
    return np.array([correct[:min(k, maxk)].reshape(-1).float().sum(0) for k in topk]).astype(np.float64)


def train_one_epoch(
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        current_epoch: int,
        total_epochs: int,
        lr_scheduler: scheduler = None,
        logger: Logger = None,
        progress: bool = False,
        mixup_fn: Optional[Mixup] = None,
        device="cuda",
):
    model.train()
    criterion.train()
    random.seed(current_epoch)

    correct_num = np.array([0., 0.])
    train_num = 0
    epoch_loss = 0
    batch_count = len(train_loader)

    if progress:
        indice = tqdm(enumerate(train_loader),
                      desc=f"train step {current_epoch}/{total_epochs}", total=len(train_loader))
    else:
        indice = enumerate(train_loader)

    start_time = time()
    for i, (images, targets) in indice:
        images = images.to(device)
        targets = targets.to(device)
        if mixup_fn is not None:
            images, soft_targets = mixup_fn(images, targets)

        output = model(images)
        if mixup_fn is not None:
            loss = criterion(output, soft_targets)
        else:
            loss = criterion(output, targets)
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            logger.critical(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step(current_epoch)

        correct_num += accuracy(output, targets, topk=(1, 5))
        train_num += targets.size(0)
        epoch_loss += loss_value

    end_time = time()
    seconds = end_time - start_time
    train_acc = correct_num / train_num
    epoch_loss /= batch_count

    log = {
        "Time_cost": seconds,
        "Loss": f"{epoch_loss:.8f}",
        "Train_Acc": train_acc
    }
    return log


def evaluate(model, test_loader, val_loader=None, test_progress=False, val_progress=False, device="cuda"):
    model.eval()
    test_correct_num, test_num = np.array([0., 0.]), 0
    if test_progress:
        indice_test = tqdm(enumerate(test_loader), desc="test step", total=len(test_loader))
    else:
        indice_test = enumerate(test_loader)

    for index, (x, label) in indice_test:
        x, label = x.to(device), label.to(device)
        pred = model(x)
        test_correct_num += accuracy(pred, label, topk=(1, 5))
        test_num += label.size(0)

    test_acc = test_correct_num / test_num
    val_acc = None
    if val_loader is not None:
        val_correct_num, val_num = np.array([0., 0.]), 0
        if val_progress:
            indice_val = tqdm(enumerate(val_loader), desc="val step", total=len(val_loader))
        else:
            indice_val = enumerate(val_loader)
        for index, (x, label) in indice_val:
            x, label = x.to(device), label.to(device)
            pred = model(x)
            val_correct_num += accuracy(pred, label, topk=(1, 5))
            val_num += label.size(0)
        val_acc = val_correct_num / val_num

    return test_acc, val_acc


def train_multi_epochs(
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        val_loader: DataLoader = None,
        criterion: nn.Module = None,
        optimizer: optim.Optimizer = None,
        lr_scheduler: scheduler = None,
        mixup_fn: Optional[Mixup] = None,
        total_epochs: int = 100,
        logger: Logger = None,
        train_progress: bool = False,
        test_progress: bool = False,
        val_progress: bool = False,
        device: str = "cuda",
        save_best: bool = True,
        model_save_path: str = None,
        draw_loss: bool = False,
        save_plot_path: str = None,
):
    logger.info("...............checking the settings...............")
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
        logger.warning(f"warning: criterion is none, automatically set it to nn.CrossEntropyLoss()")
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
        logger.warning(f"warning: optimizer is none, automatically set it to optim.Adam(model.parameters(), lr=1e-3, "
                       f"betas=(0.9, 0.999)")
    if val_loader is None:
        logger.warning("warning: val_loader is none, use test_loader instead")
    if model_save_path is None:
        model_save_path = "/home/being/fym/MambaNAS/ckpts/" + f"{type(model).__name__}.pth"
        logger.warning(f"warning: model save_path is none, automatically set it to {model_save_path}")
    if draw_loss and save_plot_path is None:
        save_plot_path = "/home/being/fym/MambaNAS/assets/" + f"{type(model).__name__}_train.png"
        logger.warning(f"warning: save_plot_path is none, automatically set it to {save_plot_path}")
    logger.info("...............checking finished...............")
    logger.info("...............start train and evaluate process...............")

    best_val_acc, best_epoch = 0, -1
    loss_history = []
    acc_history = []
    for epoch in range(total_epochs):
        log = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            current_epoch=epoch,
            total_epochs=total_epochs,
            progress=train_progress,
            device=device,
            logger=logger,
            mixup_fn=mixup_fn
        )
        test_acc, val_acc = evaluate(
            model=model,
            test_loader=test_loader,
            val_loader=val_loader,
            test_progress=test_progress,
            val_progress=val_progress,
            device=device
        )

        loss_history.append(float(log["Loss"]))
        acc_history.append(test_acc[0])

        if val_acc is None:
            val_acc = test_acc
        if save_best:
            if val_acc[0] > best_val_acc:
                best_val_acc = val_acc[0]
                best_epoch = epoch + 1
                logger.info("saving the best model")
                torch.save(model.state_dict(), model_save_path)
        else:
            torch.save(model.state_dict(), model_save_path)

        total_log = {
            "epoch": f"{epoch + 1}/{total_epochs}",
            "Time_cost": f"{log['Time_cost']:.2f}s",
            "Loss": f"{log['Loss']}",
            "Train_Acc": f"{np.round(log['Train_Acc'], decimals=4)}",
            "Test_Acc": f"{np.round(test_acc, decimals=4)}",
            "Val_Acc": f"{np.round(val_acc, decimals=4)}",
            f"Best_Val_Acc[epoch:{best_epoch}]": f"{best_val_acc:.4f}"
        }
        logger.info(total_log)

    logger.info(f"acc history:{acc_history}")
    logger.info(f"loss history:{loss_history}")
    logger.info("...............train and evaluate process finished...............")

    if draw_loss:
        epochs = np.arange(1, total_epochs + 1, 1, dtype=int).reshape(-1)
        loss_history = np.array(loss_history).reshape(-1)
        acc_history = np.array(acc_history).reshape(-1)

        fig, ax1 = plt.subplots()
        color1 = 'tab:red'
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('loss', color=color1)
        loss_line, = ax1.plot(epochs, loss_history, marker="x", label="loss", color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        ax2 = ax1.twinx()
        color2 = 'tab:blue'
        ax2.set_ylabel('acc', color=color2)
        acc_line, = ax2.plot(epochs, acc_history, marker="o", label="acc", color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        ax1.legend([loss_line, acc_line], ["loss", "acc"], loc='center right')

        fig.tight_layout()
        plt.show()

        fig.savefig(save_plot_path)
