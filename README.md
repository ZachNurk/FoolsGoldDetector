# Fools Gold Detector

This model determines whether a given image is gold or fools gold (pyrite). They both look similar due to their color, but some people still confuse the two. This model may be helpful to people trying to educate others (primarily younger audiences) about minerals, and how some minerals can look really similar to others. It may also be used by consumers to verify what they are purchasing is gold.

![gold4](https://github.com/ZachNurk/FoolsGoldDetector/assets/142443751/ca96117a-14bc-4779-b168-2433b68e489c)
![GoldProcessed](https://github.com/ZachNurk/FoolsGoldDetector/assets/142443751/60ca15e1-bd0d-45eb-a4a4-274a63ed0a7a)

## The Algorithm

The algorithm is a re-trained ResNet-18 model, fine-tuned on a large collection of Gold and Pyrite images (2 classes, see `Pyrite/labels.txt`). Weights are provided in two formats:

- `Pyrite/model_best.pth.tar` — a standard PyTorch checkpoint (`state_dict`, classes, accuracy/loss history, optimizer state). Loads with plain `torch`/`torchvision` on any OS or GPU/CPU.
- `Pyrite/resnet18.onnx` — an exported ONNX version, for inference with ONNX Runtime, or as a starting point for TensorRT engines on Jetson-class devices.

This means the model isn't tied to any particular hardware — it can be run and retrained on a regular PC, in the cloud, or on an edge device like a Jetson Nano.

## Running this project (any machine)

Requirements: Python 3.9+, `torch`, `torchvision`, `Pillow`.

```bash
pip install torch torchvision pillow
```

Run inference on an image:

```python
import torch
from torchvision import models, transforms
from PIL import Image

ckpt = torch.load("Pyrite/model_best.pth.tar", map_location="cpu", weights_only=False)
classes = ckpt["classes"]  # ['Gold', 'Pyrite']

model = models.resnet18(num_classes=ckpt["num_classes"])
model.load_state_dict(ckpt["state_dict"])
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(ckpt["resolution"]),
    transforms.CenterCrop(ckpt["resolution"]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img = Image.open("path/to/your/image.jpg").convert("RGB")
x = preprocess(img).unsqueeze(0)

with torch.no_grad():
    output = model(x)
    pred = output.argmax(1).item()

print(classes[pred])
```

## Retraining / fine-tuning further

Because `model_best.pth.tar` is a normal PyTorch checkpoint, you can resume training with any standard image-classification training loop (e.g. torchvision's `ImageFolder` + a training script), on any machine with a GPU (or CPU, just slower). At a high level:

```python
import torch
from torch import nn, optim
from torchvision import models

ckpt = torch.load("Pyrite/model_best.pth.tar", map_location="cpu", weights_only=False)

model = models.resnet18(num_classes=ckpt["num_classes"])
model.load_state_dict(ckpt["state_dict"])

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer.load_state_dict(ckpt["optimizer"])

criterion = nn.CrossEntropyLoss()
# ... plug in your DataLoader over new/expanded Gold vs. Pyrite images and continue training ...
```

Feed it more/varied Gold and Pyrite images (different lighting, backgrounds, angles) to improve generalization, then re-export to ONNX (`torch.onnx.export`) if you need the portable inference format.

## Deploying on an NVIDIA Jetson Nano (optional)

The ONNX model can also be run at the edge using NVIDIA's [jetson-inference](https://github.com/dusty-nv/jetson-inference) library and its included TensorRT-accelerated `imagenet.py` example:

1. SSH into your Jetson Nano and set up `jetson-inference`.
2. Install Python 3 and required packages:
   ```
   sudo apt-get install libpython3-dev python3-numpy
   ```
3. Place the `Pyrite` folder in `jetson-inference/python/training/classification/models`.
4. From `jetson-inference/python/training/classification`, run:
   ```
   imagenet.py --model=models/Pyrite/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=models/Pyrite/labels.txt $FILELOCATION ProcessGold.jpg
   ```
   (`$FILELOCATION` is the path to the image you want classified, e.g. `data/Pyrite/test/Gold/335.jpg`.)
5. The processed image will be written to the classification folder.

The included `resnet18.onnx.1.1.8201.GPU.FP16.engine` is a prebuilt TensorRT engine cached for a specific Jetson/TensorRT version combination — it will be regenerated automatically if versions don't match, so it's safe to delete if you hit compatibility issues.

![diagram](https://github.com/ZachNurk/FoolsGoldDetector/assets/142443751/462ea4c0-ffc9-4470-afb2-1f5f1013679d)

[Video explanation here](https://www.youtube.com/watch?v=CK6gDAKi1hE)
