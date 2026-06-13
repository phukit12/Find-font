from __future__ import annotations

import io
import json
from pathlib import Path

import streamlit as st
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps

HERE = Path(__file__).parent
MODEL_PATH = HERE / "model" / "model.ts"
LABELS_PATH = HERE / "model" / "labels.json"
TOP_K = 5

# ImageNet stats, used only if the model was trained with normalization.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

st.set_page_config(page_title="Image Classifier", page_icon="🖼️", layout="centered")


@st.cache_resource(show_spinner="Loading model…")
def load_model():
    model = torch.jit.load(str(MODEL_PATH), map_location="cpu").eval()
    meta = json.loads(LABELS_PATH.read_text())
    return model, meta["labels"], meta["preprocess"]


def preprocess(pil_img: Image.Image, pre: dict) -> torch.Tensor:
    """Reproduce the fastai validation transforms (resize-crop, scale to [0,1])."""
    size = int(pre.get("size", 224))
    # fastai "crop": scale so the shorter side == size, then center-crop size×size.
    img = TF.resize(pil_img, size)            # shorter side -> size, aspect kept
    img = TF.center_crop(img, [size, size])
    x = TF.pil_to_tensor(img).float()         # CHW, 0..255
    if pre.get("divide_255", True):
        x = x / 255.0
    if pre.get("normalize", False):
        x = TF.normalize(x, IMAGENET_MEAN, IMAGENET_STD)
    return x.unsqueeze(0)


@torch.no_grad()
def classify(model, labels, pre, pil_img: Image.Image):
    x = preprocess(pil_img, pre)
    probs = model(x).softmax(dim=1)[0]
    pairs = sorted(
        ((labels[i], float(probs[i])) for i in range(len(labels))),
        key=lambda p: p[1],
        reverse=True,
    )
    return pairs