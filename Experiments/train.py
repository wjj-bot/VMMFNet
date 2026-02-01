import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

# Import custom modules
from models.vmmfnet import VMMFNet
from utils.dataset import DualBranchDataset
from utils.loss import VMMFLoss
from utils.general import increment_path, select_device, labels_to_class_weights


def train(hyp, opt):
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)

    device = select_device(opt.device)

    # Initialize VMMFNet model
    model = VMMFNet(cfg=opt.cfg, ch=6, nc=opt.nc).to(device)

    # Optimizer and Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=hyp['lr0'], weight_decay=hyp['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs)

    # Data loaders for dual branches
    train_loader = DataLoader(
        DualBranchDataset(opt.train, opt.train_ms, imgsz=opt.imgsz, augment=True),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        DualBranchDataset(opt.val, opt.val_ms, imgsz=opt.imgsz, augment=False),
        batch_size=opt.batch_size,
        num_workers=opt.workers,
        pin_memory=True
    )

    criterion = VMMFLoss(model)
    best_fitness = 0.0

    for epoch in range(opt.epochs):
        model.train()
        mloss = torch.zeros(3, device=device)

        for i, (imgs_rgb, imgs_ms, targets, _) in enumerate(train_loader):
            imgs_rgb = imgs_rgb.to(device, non_blocking=True).float() / 255.0
            imgs_ms = imgs_ms.to(device, non_blocking=True).float() / 255.0

            # Forward pass
            pred = model(imgs_rgb, imgs_ms)
            loss, loss_items = criterion(pred, targets.to(device))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mloss = (mloss * i + loss_items) / (i + 1)

        scheduler.step()

        # Validation and Save logic
        if epoch % opt.eval_interval == 0:
            fitness = evaluate(model, val_loader, device)
            if fitness > best_fitness:
                best_fitness = fitness
                torch.save(model.state_dict(), wdir / 'best.pt')

        torch.save(model.state_dict(), wdir / 'last.pt')


def evaluate(model, dataloader, device):
    model.eval()
    # Simplified evaluation logic for mAP calculation
    return 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='models/vmmfnet.yaml')
    parser.add_argument('--train', type=str, default='data/images/train')
    parser.add_argument('--train-ms', type=str, default='data/images/train_ms')
    parser.add_argument('--val', type=str, default='data/images/val')
    parser.add_argument('--val-ms', type=str, default='data/images/val_ms')
    parser.add_argument('--nc', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--imgsz', type=int, default=512)
    parser.add_argument('--device', default='0')
    parser.add_argument('--project', default='runs/train')
    parser.add_argument('--name', default='vmmfnet_exp')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--eval-interval', type=int, default=1)
    parser.add_argument('--exist-ok', action='store_true')

    opt = parser.parse_args()

    hyp = {
        'lr0': 0.01,
        'weight_decay': 0.0005,
        'momentum': 0.937
    }

    train(hyp, opt)