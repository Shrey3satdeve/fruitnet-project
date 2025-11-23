import os
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import torch_directml as dml

from dataset import FruitQualityDataset
from model import FruitNetMultiTask
from utils import set_seed, accuracy


def make_dataloaders(data_dir: str, batch_size: int = 32, val_split: float = 0.2, seed: int = 42):
    dataset = FruitQualityDataset(data_dir)
    num_samples = len(dataset)
    indices = list(range(num_samples))
    set_seed(seed)
    import random
    random.shuffle(indices)

    split = int(num_samples * (1 - val_split))
    train_idx, val_idx = indices[:split], indices[split:]

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, dataset.fruit2idx, dataset.quality2idx


def train(args):
    device = dml.device()
    print('Using device:', device)

    train_loader, val_loader, fruit2idx, quality2idx = make_dataloaders(args.data_dir, args.batch_size, args.val_split, args.seed)

    model = FruitNetMultiTask(num_fruit_classes=len(fruit2idx), num_quality_classes=len(quality2idx))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Add learning rate scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)

    best_val = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_fruit_acc = 0.0
        running_quality_acc = 0.0
        count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]")
        for batch in pbar:
            imgs, fruit_labels, quality_labels = batch
            imgs = imgs.to(device)
            fruit_labels = fruit_labels.to(device)
            quality_labels = quality_labels.to(device)

            optimizer.zero_grad()
            fruit_logits, quality_logits = model(imgs)

            loss_fruit = criterion(fruit_logits, fruit_labels)
            loss_quality = criterion(quality_logits, quality_labels)
            loss = loss_fruit + loss_quality

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_fruit_acc += accuracy(fruit_logits.detach().cpu(), fruit_labels.detach().cpu())
            running_quality_acc += accuracy(quality_logits.detach().cpu(), quality_labels.detach().cpu())
            count += 1

            pbar.set_postfix({'loss': running_loss / count, 'fruit_acc': running_fruit_acc / count, 'quality_acc': running_quality_acc / count})

        # Validation
        model.eval()
        val_loss = 0.0
        val_f_acc = 0.0
        val_q_acc = 0.0
        vcount = 0
        with torch.no_grad():
            for imgs, fruit_labels, quality_labels in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]"):
                imgs = imgs.to(device)
                fruit_labels = fruit_labels.to(device)
                quality_labels = quality_labels.to(device)

                fruit_logits, quality_logits = model(imgs)
                loss = criterion(fruit_logits, fruit_labels) + criterion(quality_logits, quality_labels)

                val_loss += loss.item()
                val_f_acc += accuracy(fruit_logits.detach().cpu(), fruit_labels.detach().cpu())
                val_q_acc += accuracy(quality_logits.detach().cpu(), quality_labels.detach().cpu())
                vcount += 1

        avg_val_loss = val_loss / max(1, vcount)
        avg_val_f_acc = val_f_acc / max(1, vcount)
        avg_val_q_acc = val_q_acc / max(1, vcount)

        print(f"Epoch {epoch}: Train loss {running_loss/count:.4f} | Val loss {avg_val_loss:.4f}")
        print(f"Fruit val acc: {avg_val_f_acc:.4f} | Quality val acc: {avg_val_q_acc:.4f}")

        # Save best model by val loss
        if avg_val_loss < best_val:
            best_val = avg_val_loss
            save_path = Path(args.save_path)
            state = {
                'model_state_dict': model.state_dict(),
                'fruit2idx': fruit2idx,
                'quality2idx': quality2idx,
                'epoch': epoch,
            }
            torch.save(state, save_path)
            print(f"Saved best model to {save_path} (val_loss={best_val:.4f})")
        
        # Step the scheduler
        scheduler.step()
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")


def parse_args():
    parser = argparse.ArgumentParser(description='Train FruitNet Multi-task')
    parser.add_argument('--data-dir', type=str, default=r'D:\archive\Processed Images_Fruits\Mixed Qualit_Fruits', help='Base dataset directory')
    parser.add_argument('--epochs', type=int, default=8)  # Increase default epochs
    parser.add_argument('--batch-size', type=int, default=32)  # Increase default batch size
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--val-split', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-path', type=str, default='fruitnet_multitask.pth')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print('Args:', args)
    train(args)
