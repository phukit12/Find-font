from __future__ import annotations

import io, base64, json
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torchvision.models as models
from PIL import Image, ImageOps

HERE = Path(__file__).parent
WEIGHTS_BODY_PATH = HERE / "model" / "weights_body.pth"
WEIGHTS_HEAD_PATH = HERE / "model" / "weights_head.pth"
LABELS_PATH       = HERE / "model" / "labels.json"
TOP_K = 5

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

st.set_page_config(page_title="Thai Font Finder", page_icon="🔍", layout="wide")

st.html("""
<link href="https://fonts.googleapis.com/css2?family=Sarabun:wght@400;500;600&display=swap" rel="stylesheet">
<style>
#MainMenu, footer, header { visibility: hidden; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
section.main, .main .block-container {
    background: #f0ead8 !important;
    color: #2c2c2a !important;
    font-family: 'Sarabun', sans-serif !important;
}
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── Style the native file uploader to look like ref ── */
[data-testid="stFileUploader"] {
    background: #ffffff !important;
    border-bottom: 0.5px solid #d6cebc !important;
    border-radius: 0 !important;
    padding: 6px 1.2rem !important;
}
[data-testid="stFileUploader"] label {
    display: none !important;
}
[data-testid="stFileUploader"] section {
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
}
[data-testid="stFileUploader"] section > div {
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
}
[data-testid="stFileUploadDropzone"] {
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    flex-direction: row !important;
    gap: 10px !important;
}
[data-testid="stFileUploadDropzone"] > div {
    display: flex !important;
    align-items: center !important;
    flex-direction: row !important;
    gap: 10px !important;
}
/* Upload button */
[data-testid="stFileUploadDropzone"] button {
    background: #f0ead8 !important;
    border: 0.5px solid #b4b2a9 !important;
    border-radius: 6px !important;
    color: #3b3a37 !important;
    font-family: 'Sarabun', sans-serif !important;
    font-size: 13px !important;
    padding: 5px 14px !important;
    white-space: nowrap !important;
}
[data-testid="stFileUploadDropzone"] button:hover {
    background: #e8e2d0 !important;
}
/* hint text */
[data-testid="stFileUploadDropzone"] span,
[data-testid="stFileUploadDropzone"] p {
    font-size: 12px !important;
    color: #888780 !important;
    font-family: 'Sarabun', sans-serif !important;
}

/* ── Header ── */
.tff-header {
    position: relative; background: #f0ead8;
    padding: 1.6rem 1rem 1.3rem; text-align: center;
    overflow: hidden; min-height: 130px;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    border-bottom: 0.5px solid #d6cebc;
}
.tff-hbg {
    position: absolute; inset: 0; overflow: hidden;
    display: flex; flex-direction: column; justify-content: center;
    pointer-events: none;
}
.tff-hbg-row {
    white-space: nowrap; font-size: 30px;
    font-family: 'Sarabun', sans-serif;
    color: rgba(80,60,20,0.13); line-height: 1.6; letter-spacing: 8px;
}
.tff-logo {
    position: relative; z-index: 2;
    display: flex; align-items: center; justify-content: center; gap: 8px;
}
.tff-logo-kor  { font-size: 48px; color: #2d5a3d; font-weight: 600; line-height: 1; }
.tff-logo-text { font-size: 28px; font-weight: 600; color: #2c2c2a; letter-spacing: 1px; }
.tff-logo-a    { font-size: 38px; font-weight: 600; color: #2c2c2a; }
.tff-subtitle  {
    position: relative; z-index: 2; margin-top: 8px;
    background: #d6cebc; color: #4a4640; font-size: 13px;
    padding: 3px 18px; border-radius: 20px; display: inline-block;
    font-family: 'Sarabun', sans-serif;
}

/* ── Content grid ── */
.tff-content {
    display: grid; grid-template-columns: 1fr 1.4fr;
    gap: 1rem; padding: 1rem 1.2rem; background: #f0ead8;
}
.tff-preview-card {
    border: 1.5px dashed #b4b2a9; border-radius: 10px;
    background: rgba(255,255,255,0.35); padding: 0.85rem 1rem;
    min-height: 230px; display: flex; flex-direction: column; align-items: center;
}
.tff-preview-label {
    font-size: 13px; color: #5f5e5a; font-weight: 500;
    font-family: 'Sarabun', sans-serif; margin-bottom: 12px; align-self: flex-start;
}
.tff-preview-img-wrap {
    flex: 1; display: flex; align-items: center; justify-content: center; width: 100%;
}
.tff-preview-img-wrap img { max-width:100%; max-height:180px; border-radius:8px; object-fit:contain; }
.tff-paper {
    width: 110px; height: 100px; background: #e8e4d8;
    border-radius: 6px; position: relative;
}
.tff-paper::after {
    content: ''; position: absolute; top: 0; right: 0;
    width: 20px; height: 20px; background: #f0ead8;
    clip-path: polygon(0 0, 100% 100%, 100% 0);
}
.tff-results-title {
    font-size: 17px; font-weight: 600; color: #2c2c2a;
    margin-bottom: 12px; font-family: 'Sarabun', sans-serif;
}
.tff-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; margin-bottom: 8px; }
.tff-card {
    background: rgba(255,255,255,0.55); border: 0.5px solid #d3d1c7;
    border-radius: 10px; padding: 10px 12px; font-family: 'Sarabun', sans-serif;
}
.tff-card.top { background: rgba(200,230,210,0.65); border: 1px solid #5a9a6a; }
.tff-card-top { display: flex; align-items: flex-start; justify-content: space-between; margin-bottom: 2px; }
.tff-rank     { font-size: 18px; font-weight: 500; color: #888780; }
.tff-name     { font-size: 13px; font-weight: 500; color: #2c2c2a; margin-bottom: 4px; }
.tff-pct-high { font-size: 14px; font-weight: 600; color: #2d5a3d; }
.tff-pct-mid  { font-size: 13px; font-weight: 600; color: #b07d10; }
.tff-pct-low  { font-size: 13px; color: #888780; }
.tff-bar-bg   { height: 3px; background: #d3d1c7; border-radius: 2px; margin-top: 4px; }
.tff-bar      { height: 3px; border-radius: 2px; }
.tff-bar-green  { background: #3b7a50; }
.tff-bar-yellow { background: #d4a017; }
.tff-bar-gray   { background: #b4b2a9; }
.tff-empty { color: #888780; font-size: 14px; font-family: 'Sarabun', sans-serif; }
</style>
""")

# ── Header ──────────────────────────────────────────────────────────────
st.html("""
<div class="tff-header">
    <div class="tff-hbg">
        <div class="tff-hbg-row">ก ข ฃ ค ฅ ฆ ง จ ฉ ช ซ ฌ ญ ฎ ฏ ฐ ฑ ฒ ณ ด ต ถ ท ธ น บ ป ผ ฝ พ ฟ ภ ม ย ร ล ว ศ ษ ส ห ฬ อ ฮ</div>
        <div class="tff-hbg-row">ะ า ิ ี ึ ื ุ ู เ แ โ ใ ไ ็ ่ ้ ๊ ๋ ์ ๆ ฯ ๐ ๑ ๒ ๓ ๔ ๕ ๖ ๗ ๘ ๙ ก ข ค ง จ ช ซ ญ ด ต ถ ท น บ</div>
        <div class="tff-hbg-row">พ ฟ ม ย ร ล ว ส ห อ ฮ ะ า เ แ โ ใ ไ ก ข ฃ ค ฅ ฆ ง จ ฉ ช ซ ฌ ญ ฎ ฏ ฐ ฑ ฒ ณ ด ต ถ ท ธ น</div>
    </div>
    <div class="tff-logo">
        <span class="tff-logo-kor">ก</span>
        <span style="font-size:38px;color:#2d5a3d;margin-left:-8px;">🔍</span>
        <span class="tff-logo-text">Thai Font Finder</span>
        <span class="tff-logo-a">A</span>
    </div>
    <div class="tff-subtitle">โปรแกรมค้นหาฟอนต์ภาษาไทย</div>
</div>
""")

# ── Native file uploader (functional + styled) ──────────────────────────
uploaded_file = st.file_uploader(
    "นำเข้ารูปภาพ",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    label_visibility="collapsed",
)

# ── Model ────────────────────────────────────────────────────────────────
class AdaptiveConcatPool2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool2d(1)
        self.mp = nn.AdaptiveMaxPool2d(1)
    def forward(self, x):
        return torch.cat([self.ap(x), self.mp(x)], 1)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

@st.cache_resource(show_spinner="กำลังโหลดโมเดล…")
def load_model():
    meta = json.loads(LABELS_PATH.read_text())
    num_classes = len(meta["labels"])
    body = models.resnet50()
    body = nn.Sequential(*list(body.children())[:-2])
    body.load_state_dict(torch.load(WEIGHTS_BODY_PATH, map_location="cpu"), strict=False)
    head = nn.Sequential(
        AdaptiveConcatPool2d(), Flatten(),
        nn.BatchNorm1d(4096), nn.Dropout(p=0.25),
        nn.Linear(4096, 512, bias=False), nn.ReLU(inplace=True),
        nn.BatchNorm1d(512), nn.Dropout(p=0.5),
        nn.Linear(512, num_classes, bias=False),
    )
    head.load_state_dict(torch.load(WEIGHTS_HEAD_PATH, map_location="cpu"), strict=False)
    return nn.Sequential(body, head).eval(), meta["labels"], meta["preprocess"]

def preprocess(pil_img, pre):
    size = int(pre.get("size", 224))
    img = TF.resize(pil_img, size)
    img = TF.center_crop(img, [size, size])
    x = TF.pil_to_tensor(img).float()
    if pre.get("divide_255", True): x = x / 255.0
    if pre.get("normalize", False): x = TF.normalize(x, IMAGENET_MEAN, IMAGENET_STD)
    return x.unsqueeze(0)

@torch.no_grad()
def classify(model, labels, pre, pil_img):
    probs = model(preprocess(pil_img, pre)).softmax(dim=1)[0]
    return sorted(
        ((labels[i], float(probs[i])) for i in range(len(labels))),
        key=lambda p: p[1], reverse=True,
    )

try:
    model, labels, pre_config = load_model()
except Exception as e:
    st.error(f"ไม่สามารถโหลดโมเดลได้: {e}")
    st.stop()

# ── Build + render content ───────────────────────────────────────────────
image = None
if uploaded_file:
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image)
    if image.mode != "RGB":
        image = image.convert("RGB")

if image:
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    preview_html = f'<img src="data:image/jpeg;base64,{b64}" alt="preview">'
else:
    preview_html = '<div class="tff-paper"></div>'

def pct_cls(p):
    return "tff-pct-high" if p >= 0.85 else ("tff-pct-mid" if p >= 0.60 else "tff-pct-low")
def bar_cls(p):
    return "tff-bar-green" if p >= 0.85 else ("tff-bar-yellow" if p >= 0.60 else "tff-bar-gray")
def card(rank, label, prob, extra=""):
    bw = int(prob * 100)
    return f"""<div class="tff-card {extra}">
        <div class="tff-card-top">
            <span class="tff-rank">{rank}</span>
            <span class="{pct_cls(prob)}">{prob*100:.1f}%</span>
        </div>
        <div class="tff-name">{label}</div>
        <div class="tff-bar-bg"><div class="tff-bar {bar_cls(prob)}" style="width:{bw}%"></div></div>
    </div>"""

if image:
    with st.spinner("กำลังวิเคราะห์..."):
        results = classify(model, labels, pre_config, image)
    top5 = results[:TOP_K]
    row1 = "".join(card(i+1, l, p, "top" if i==0 else "") for i,(l,p) in enumerate(top5[:3]))
    row2 = "".join(card(i+4, l, p) for i,(l,p) in enumerate(top5[3:]))
    row2 += '<div class="tff-card" style="background:transparent;border:none;"></div>' * (3 - len(top5[3:]))
    right_html = f"""
        <div class="tff-results-title">ผลการพยากรณ์</div>
        <div class="tff-grid">{row1}</div>
        <div class="tff-grid">{row2}</div>"""
else:
    right_html = """
        <div class="tff-results-title">ผลการพยากรณ์</div>
        <p class="tff-empty">อัปโหลดรูปภาพเพื่อดูผลการพยากรณ์</p>"""

st.html(f"""
<div class="tff-content">
    <div>
        <div class="tff-preview-card">
            <div class="tff-preview-label">พรีวิวรูปภาพที่อัปโหลด</div>
            <div class="tff-preview-img-wrap">{preview_html}</div>
        </div>
    </div>
    <div>{right_html}</div>
</div>
""")
