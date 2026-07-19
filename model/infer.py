"""Classify a single image as Gold or Pyrite using model/checkpoints/model_best.pth.tar.

Usage:
    python3 model/infer.py path/to/image.jpg
"""
import argparse

from PIL import Image

from common import load_model, predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--checkpoint", default="model/checkpoints/model_best.pth.tar")
    args = parser.parse_args()

    model, classes, resolution = load_model(args.checkpoint)
    img = Image.open(args.image)
    probs = predict(model, classes, resolution, img)
    top_class = max(probs, key=probs.get)

    print(f"{top_class} ({probs[top_class] * 100:.1f}% confidence)")
    for cls, p in probs.items():
        print(f"  {cls}: {p * 100:.1f}%")


if __name__ == "__main__":
    main()
