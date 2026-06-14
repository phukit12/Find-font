from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
import torch
import torchvision.transforms.functional as TF
import torchvision.models as models
from PIL import Image, ImageOps

HERE = Path(__file__).parent
WEIGHTS_PATH = HERE / "model" / "weights.pth"
LABELS_PATH = HERE / "model" / "labels.json"
TOP_K = 5

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

st.set_page_config(page_title="Image Classifier", page_icon="🖼️", layout="centered")


@st.cache_resource(show_spinner="Loading model…")
def load_model():
    meta = json.loads(LABELS_PATH.read_text())
    num_classes = len(meta["labels"])

    model = models.resnet50()
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"), strict=False)
    model.eval()

    return model, meta["labels"], meta["preprocess"]


def preprocess(pil_img: Image.Image, pre: dict) -> torch.Tensor:
    size = int(pre.get("size", 224))
    img = TF.resize(pil_img, size)
    img = TF.center_crop(img, [size, size])
    x = TF.pil_to_tensor(img).float()
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


try:
    model, labels, pre_config = load_model()
except Exception as e:
    st.error(f"ไม่สามารถโหลดโมเดลได้: {e}")
    st.stop()

st.title("🖼️ แอปพลิเคชันจำแนกรูปภาพ")
st.write("อัปโหลดรูปภาพของคุณด้านล่าง เพื่อให้ AI ช่วยวิเคราะห์")

uploaded_file = st.file_uploader("เลือกรูปภาพ...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image)

    st.image(image, caption="รูปภาพที่อัปโหลด", use_container_width=True)

    with st.spinner("กำลังวิเคราะห์รูปภาพ..."):
        results = classify(model, labels, pre_config, image)

    st.subheader("ผลการวิเคราะห์ (Top 5)")
    for label, prob in results[:TOP_K]:
        percent = prob * 100
        st.write(f"**{label}**: {percent:.2f}%")
        st.progress(prob)