"""Shared checkpoint-loading and prediction logic for infer.py and app.py."""
import torch
from torchvision import models, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_model(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = models.resnet18(num_classes=ckpt["num_classes"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["classes"], ckpt["resolution"]


def build_preprocess(resolution):
    return transforms.Compose([
        transforms.Resize(resolution),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def predict(model, classes, resolution, pil_image):
    """Returns {class_name: probability} for a PIL image."""
    preprocess = build_preprocess(resolution)
    x = preprocess(pil_image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    return {cls: probs[i].item() for i, cls in enumerate(classes)}
