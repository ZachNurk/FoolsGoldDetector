"""Export model/checkpoints/model_best.pth.tar to ONNX for portable/edge inference.

Usage:
    python3 model/export_onnx.py
"""
import argparse

import torch

from common import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="model/checkpoints/model_best.pth.tar")
    parser.add_argument("--output", default="model/checkpoints/resnet18.onnx")
    args = parser.parse_args()

    model, _, resolution = load_model(args.checkpoint)

    dummy = torch.randn(1, 3, resolution, resolution)
    torch.onnx.export(
        model, dummy, args.output,
        input_names=["input_0"], output_names=["output_0"],
        dynamic_axes={"input_0": {0: "batch"}, "output_0": {0: "batch"}},
        opset_version=12,
    )
    print(f"Exported {args.output}")


if __name__ == "__main__":
    main()
