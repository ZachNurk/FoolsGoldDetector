# Fools Gold Detector

This model determines whether a given image is gold or fools gold (pyrite). They both look similar due to their color, but some people still confuse the two. This model may be helpful to people trying to educate others (primarily younger audiences) about minerals, and how some minerals can look really similar to others. It may also be used by consumers to verify what they are purchasing is gold.

![gold4](https://github.com/ZachNurk/FoolsGoldDetector/assets/142443751/ca96117a-14bc-4779-b168-2433b68e489c)
![GoldProcessed](https://github.com/ZachNurk/FoolsGoldDetector/assets/142443751/60ca15e1-bd0d-45eb-a4a4-274a63ed0a7a)

## The Algorithm

The algorithm is a re-trained ResNet-18 model, fine-tuned on a collection of Gold and Pyrite images (2 classes, see `model/checkpoints/labels.txt`). Weights are provided in two formats:

- `model/checkpoints/model_best.pth.tar` — a standard PyTorch checkpoint (`state_dict`, classes, accuracy/loss history, optimizer state). Loads with plain `torch`/`torchvision` on any OS or GPU/CPU.
- `model/checkpoints/resnet18.onnx` — an exported ONNX version, for inference with ONNX Runtime, or as a starting point for TensorRT engines on Jetson-class devices.

This means the model isn't tied to any particular hardware — it can be run and retrained on a regular PC, in the cloud, or on an edge device like a Jetson Nano.

## Repo layout

```
model/                     everything related to training/inference
  data/
    train/{Gold,Pyrite}/   training images
    val/{Gold,Pyrite}/     validation images (used to pick the best checkpoint)
    test/{Gold,Pyrite}/    held-out images for final evaluation
  checkpoints/
    model_best.pth.tar      best checkpoint so far (by val accuracy)
    resnet18.onnx           ONNX export of model_best
    labels.txt              class names, one per line
  common.py                shared checkpoint-loading/inference logic
  train.py                 train / resume-train on data/
  infer.py                 classify a single image (CLI)
  export_onnx.py           re-export a checkpoint to ONNX
  requirements.txt
app/                        everything related to the web UI
  package.json              npm run dev runs backend + frontend together
  backend/
    main.py                 FastAPI service exposing POST /api/predict
    requirements.txt
  frontend/                 Vite + React + TypeScript web UI
```

`model/data/*/Gold` and `model/data/*/Pyrite` currently only contain `.gitkeep` placeholders — drop your own images in (standard `torchvision.datasets.ImageFolder` layout: one subfolder per class) before training.

## Setup

```bash
pip install -r model/requirements.txt
```

## Running inference

```bash
python3 model/infer.py path/to/your/image.jpg
```

## Web UI

A FastAPI backend serves the model, and a Vite/React/TypeScript frontend lets you drag-and-drop an image in the browser and see the Gold/Pyrite confidence breakdown.

One-time setup:
```bash
pip install -r app/backend/requirements.txt
cd app && npm install            # installs concurrently
npm --prefix frontend install
```

Then, from `app/`:
```bash
npm run dev
```

This runs the FastAPI server (port 8000) and the Vite dev server (port 5173) together. Open http://localhost:5173 — the Vite dev server proxies `/api/*` requests to the FastAPI server.

## Training / retraining

Populate `model/data/train/Gold`, `model/data/train/Pyrite`, `model/data/val/Gold`, `model/data/val/Pyrite` with images, then:

```bash
# train from scratch
python3 model/train.py --data-dir model/data --epochs 10

# fine-tune further from the existing checkpoint
python3 model/train.py --data-dir model/data --epochs 10 --resume model/checkpoints/model_best.pth.tar
```

Each epoch writes `model/checkpoints/checkpoint.pth.tar` (latest) and, when validation accuracy improves, `model/checkpoints/model_best.pth.tar` (best-so-far) — the same checkpoint format used for inference and export.

To refresh the ONNX export after retraining:

```bash
python3 model/export_onnx.py
```

## Deploying on an NVIDIA Jetson Nano (optional)

The ONNX model can also be run at the edge using NVIDIA's [jetson-inference](https://github.com/dusty-nv/jetson-inference) library and its included TensorRT-accelerated `imagenet.py` example:

1. SSH into your Jetson Nano and set up `jetson-inference`.
2. Install Python 3 and required packages:
   ```
   sudo apt-get install libpython3-dev python3-numpy
   ```
3. Place the `model/checkpoints` folder (renamed to e.g. `Pyrite`) under `jetson-inference/python/training/classification/models`.
4. From `jetson-inference/python/training/classification`, run:
   ```
   imagenet.py --model=models/Pyrite/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=models/Pyrite/labels.txt $FILELOCATION ProcessGold.jpg
   ```
   (`$FILELOCATION` is the path to the image you want classified, e.g. `model/data/test/Gold/example.jpg`.)
5. The processed image will be written to the classification folder.

Note: `jetson-inference` will generate its own cached TensorRT engine file (`*.engine`) next to the ONNX model on first run — that file is specific to your device's GPU/TensorRT version and doesn't need to be committed to this repo.

![diagram](https://github.com/ZachNurk/FoolsGoldDetector/assets/142443751/462ea4c0-ffc9-4470-afb2-1f5f1013679d)

[Video explanation here](https://www.youtube.com/watch?v=CK6gDAKi1hE)
