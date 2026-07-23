"""Evaluate model/checkpoints/model_best.pth.tar against a held-out test set.

Usage:
    python3 model/test.py --data-dir model/data/test
"""
import argparse
from collections import defaultdict
from pathlib import Path

from PIL import Image

from common import load_model, predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="model/data/test")
    parser.add_argument("--checkpoint", default="model/checkpoints/model_best.pth.tar")
    args = parser.parse_args()

    model, classes, resolution = load_model(args.checkpoint)

    confusion = defaultdict(lambda: defaultdict(int))
    total = correct = 0

    for true_cls in classes:
        class_dir = Path(args.data_dir) / true_cls
        if not class_dir.is_dir():
            continue
        for img_path in sorted(class_dir.iterdir()):
            if img_path.name.startswith("."):
                continue
            img = Image.open(img_path)
            img.load()
            probs = predict(model, classes, resolution, img)
            pred_cls = max(probs, key=probs.get)
            confusion[true_cls][pred_cls] += 1
            total += 1
            correct += pred_cls == true_cls

    print(f"overall accuracy: {correct}/{total} = {correct / total:.4f}\n")
    print(f"{'true \\ pred':<12}" + "".join(f"{c:<10}" for c in classes))
    for true_cls in classes:
        row_total = sum(confusion[true_cls].values())
        row_correct = confusion[true_cls][true_cls]
        acc = row_correct / row_total if row_total else 0.0
        print(f"{true_cls:<12}" + "".join(f"{confusion[true_cls][c]:<10}" for c in classes) + f"  acc={acc:.4f}")


if __name__ == "__main__":
    main()
