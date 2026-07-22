"""Train or resume-train the Gold/Pyrite ResNet-18 classifier.

Expects data/train and data/val, each containing one subfolder per class
(Gold/, Pyrite/) of images (torchvision.datasets.ImageFolder layout).

Usage:
    python3 model/train.py --data-dir model/data --epochs 10
    python3 model/train.py --data-dir model/data --epochs 10 --resume model/checkpoints/model_best.pth.tar
"""
import argparse
import os
import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

RESOLUTION = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(train):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(RESOLUTION),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(RESOLUTION),
        transforms.CenterCrop(RESOLUTION),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def run_epoch(model, loader, criterion, optimizer, device, train, epoch, log_every=5):
    model.train() if train else model.eval()
    total_loss, correct, count = 0.0, 0, 0
    phase = "train" if train else "val"
    num_batches = len(loader)
    start = time.time()
    with torch.set_grad_enabled(train):
        for batch_idx, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            batch_loss = loss.item()
            batch_correct = (outputs.argmax(1) == labels).sum().item()
            total_loss += batch_loss * images.size(0)
            correct += batch_correct
            count += images.size(0)
            if batch_idx % log_every == 0 or batch_idx == num_batches - 1:
                elapsed = time.time() - start
                print(f"  epoch {epoch} [{phase}] batch {batch_idx + 1}/{num_batches} "
                      f"loss={batch_loss:.4f} acc={batch_correct / images.size(0):.4f} "
                      f"elapsed={elapsed:.1f}s", flush=True)
    return total_loss / count, correct / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="model/data")
    parser.add_argument("--model-dir", default="model/checkpoints")
    parser.add_argument("--resume", default=None, help="checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    train_ds = datasets.ImageFolder(os.path.join(args.data_dir, "train"), build_transforms(True))
    val_ds = datasets.ImageFolder(os.path.join(args.data_dir, "val"), build_transforms(False))
    classes = train_ds.classes
    print(f"classes: {classes}", flush=True)
    print(f"train images: {len(train_ds)}, val images: {len(val_ds)}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    print("building model (ImageNet-pretrained)...", flush=True)
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    start_epoch = 0
    best_acc = 0.0

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        best_acc = ckpt.get("accuracy", {}).get("val", 0.0)
        classes = ckpt.get("classes", classes)
        print(f"Resumed from {args.resume} at epoch {start_epoch}, best val acc {best_acc:.4f}")

    model.to(device)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(args.model_dir, exist_ok=True)

    print(f"starting training: epochs {start_epoch}..{start_epoch + args.epochs - 1}, "
          f"batch_size={args.batch_size}, lr={args.lr}", flush=True)

    for epoch in range(start_epoch, start_epoch + args.epochs):
        epoch_start = time.time()
        print(f"== epoch {epoch} start ==", flush=True)
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True, epoch=epoch)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False, epoch=epoch)
        epoch_time = time.time() - epoch_start
        print(f"epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} time={epoch_time:.1f}s", flush=True)

        checkpoint = {
            "epoch": epoch + 1,
            "arch": "resnet18",
            "resolution": RESOLUTION,
            "classes": classes,
            "num_classes": len(classes),
            "multi_label": False,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "accuracy": {"train": train_acc, "val": val_acc},
            "loss": {"train": train_loss, "val": val_loss},
        }
        torch.save(checkpoint, os.path.join(args.model_dir, "checkpoint.pth.tar"))
        print(f"  saved checkpoint.pth.tar", flush=True)
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(checkpoint, os.path.join(args.model_dir, "model_best.pth.tar"))
            print(f"  -> new best (val_acc={val_acc:.4f}), saved model_best.pth.tar", flush=True)


if __name__ == "__main__":
    main()
