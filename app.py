from __future__ import annotations

import json
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

# ═══════════════════════════════════════════════════════════════════════
#  CSS
# ═══════════════════════════════════════════════════════════════════════
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Sarabun:wght@400;500;600&display=swap" rel="stylesheet">
<style>

/* ── Reset Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}
[data-testid="stAppViewContainer"] {
    background: #f0ead8;
    font-family: 'Sarabun', sans-serif;
}

/* hide default file-uploader widget entirely — we render our own bar */
[data-testid="stFileUploader"] { display: none !important; }

/* ── Header ── */
.tff-header {
    position: relative;
    background: #f0ead8;
    padding: 1.6rem 1rem 1.3rem;
    text-align: center;
    overflow: hidden;
    min-height: 130px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    border-bottom: 0.5px solid #d6cebc;
}
.tff-hbg {
    position: absolute; inset: 0;
    overflow: hidden;
    display: flex; flex-direction: column; justify-content: center;
    pointer-events: none;
}
.tff-hbg-row {
    white-space: nowrap;
    font-size: 30px; font-family: 'Sarabun', sans-serif;
    color: rgba(80,60,20,0.13);
    line-height: 1.6; letter-spacing: 8px;
}
.tff-logo {
    position: relative; z-index: 2;
    display: flex; align-items: center; justify-content: center; gap: 8px;
    font-family: 'Sarabun', sans-serif;
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

/* ── Upload bar (full-width) ── */
.tff-upload-bar {
    background: #fff;
    border-bottom: 0.5px solid #d6cebc;
    padding: 8px 1.2rem;
    display: flex; align-items: center; gap: 10px;
    cursor: pointer;
}
.tff-upload-btn {
    background: #f0ead8;
    border: 0.5px solid #b4b2a9;
    border-radius: 6px;
    padding: 5px 14px;
    font-size: 13px; color: #3b3a37;
    display: flex; align-items: center; gap: 6px;
    white-space: nowrap;
    font-family: 'Sarabun', sans-serif;
    cursor: pointer;
}
.tff-upload-hint {
    font-size: 12px; color: #888780; flex: 1;
    font-family: 'Sarabun', sans-serif;
}
.tff-upload-arrow { font-size: 16px; color: #888780; }

/* ── Content area ── */
.tff-content {
    display: grid;
    grid-template-columns: 1fr 1.4fr;
    gap: 1rem;
    padding: 1rem 1.2rem;
    background: #f0ead8;
}

/* ── Preview card (left) ── */
.tff-preview-card {
    border: 1.5px dashed #b4b2a9;
    border-radius: 10px;
    background: rgba(255,255,255,0.35);
    padding: 0.85rem 1rem;
    min-height: 230px;
    display: flex; flex-direction: column; align-items: center;
}
.tff-preview-label {
    font-size: 13px; color: #5f5e5a; font-weight: 500;
    font-family: 'Sarabun', sans-serif;
    margin-bottom: 12px; align-self: flex-start;
}
.tff-preview-img-wrap {
    flex: 1; display: flex; align-items: center; justify-content: center;
    width: 100%;
}
.tff-preview-img-wrap img {
    max-width: 100%; max-height: 180px;
    border-radius: 8px; object-fit: contain;
}
/* paper placeholder */
.tff-paper {
    width: 110px; height: 100px;
    background: #e8e4d8; border-radius: 6px;
    position: relative;
}
.tff-paper::after {
    content: '';
    position: absolute; top: 0; right: 0;
    width: 20px; height: 20px;
    background: #f0ead8;
    clip-path: polygon(0 0, 100% 100%, 100% 0);
}

/* ── Results (right) ── */
.tff-results-title {
    font-size: 17px; font-weight: 600; color: #2c2c2a;
    margin-bottom: 12px;
    font-family: 'Sarabun', sans-serif;
}
.tff-grid {
    display: grid; grid-template-columns: 1fr 1fr 1fr;
    gap: 8px; margin-bottom: 8px;
}
.tff-card {
    background: rgba(255,255,255,0.55);
    border: 0.5px solid #d3d1c7;
    border-radius: 10px; padding: 10px 12px;
    font-family: 'Sarabun', sans-serif;
}
.tff-card.top {
    background: rgba(200,230,210,0.65);
    border: 1px solid #5a9a6a;
}
.tff-card-top {
    display: flex; align-items: flex-start;
    justify-content: space-between; margin-bottom: 2px;
}
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
.tff-empty {
    color: #888780; font-size: 14px;
    font-family: 'Sarabun', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
#  Header
# ═══════════════════════════════════════════════════════════════════════
st.markdown("""
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
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
#  Upload bar (visual only — triggers the hidden Streamlit widget)
# ═══════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="tff-upload-bar" onclick="document.querySelector('[data-testid=stFileUploadDropzone]')?.click()">
    <div class="tff-upload-btn">⬆ นำเข้ารูปภาพ</div>
    <span class="tff-upload-hint">Choose an image — PNG, JPG, WEBP, BMP สูงสุด 20MB</span>
    <span class="tff-upload-arrow">⌄</span>
</div>
""", unsafe_allow_html=True)

# hidden but functional uploader
uploaded_file = st.file_uploader(
    "upload", type=["jpg", "jpeg", "png", "webp", "bmp"],
    label_visibility="collapsed",
)

# ═══════════════════════════════════════════════════════════════════════
#  Model
# ═══════════════════════════════════════════════════════════════════════
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
    model = nn.Sequential(body, head).eval()
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
    return sorted(
        ((labels[i], float(probs[i])) for i in range(len(labels))),
        key=lambda p: p[1], reverse=True,
    )

try:
    model, labels, pre_config = load_model()
except Exception as e:
    st.error(f"ไม่สามารถโหลดโมเดลได้: {e}")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════
#  Content: build HTML block in one shot → single st.markdown call
# ═══════════════════════════════════════════════════════════════════════
image = None
if uploaded_file:
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image)
    if image.mode != "RGB":
        image = image.convert("RGB")

# ── left: preview ──
if image:
    import io, base64
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    preview_html = f'<img src="data:image/jpeg;base64,{b64}" alt="preview">'
else:
    preview_html = '<div class="tff-paper"></div>'

# ── right: result cards ──
def pct_class(p):
    if p >= 0.85: return "tff-pct-high"
    if p >= 0.60: return "tff-pct-mid"
    return "tff-pct-low"

def bar_class(p):
    if p >= 0.85: return "tff-bar-green"
    if p >= 0.60: return "tff-bar-yellow"
    return "tff-bar-gray"

if image:
    with st.spinner("กำลังวิเคราะห์..."):
        results = classify(model, labels, pre_config, image)
    top5 = results[:TOP_K]

    def card(rank, label, prob, extra_cls=""):
        pct_str = f"{prob*100:.1f}%"
        bar_w   = int(prob * 100)
        return f"""
        <div class="tff-card {extra_cls}">
            <div class="tff-card-top">
                <span class="tff-rank">{rank}</span>
                <span class="{pct_class(prob)}">{pct_str}</span>
            </div>
            <div class="tff-name">{label}</div>
            <div class="tff-bar-bg">
                <div class="tff-bar {bar_class(prob)}" style="width:{bar_w}%"></div>
            </div>
        </div>"""

    row1 = "".join(card(i+1, l, p, "top" if i==0 else "") for i,(l,p) in enumerate(top5[:3]))
    row2_cards = "".join(card(i+4, l, p) for i,(l,p) in enumerate(top5[3:]))
    row2_pad   = '<div class="tff-card" style="background:transparent;border:none;"></div>' * (3 - len(top5[3:]))
    row2 = row2_cards + row2_pad

    results_html = f"""
    <div class="tff-results-title">ผลการพยากรณ์</div>
    <div class="tff-grid">{row1}</div>
    <div class="tff-grid">{row2}</div>
    """
else:
    results_html = """
    <div class="tff-results-title">ผลการพยากรณ์</div>
    <p class="tff-empty">อัปโหลดรูปภาพเพื่อดูผลการพยากรณ์</p>
    """

st.markdown(f"""
<div class="tff-content">
    <div>
        <div class="tff-preview-card">
            <div class="tff-preview-label">พรีวิวรูปภาพที่อัปโหลด</div>
            <div class="tff-preview-img-wrap">{preview_html}</div>
        </div>
    </div>
    <div>{results_html}</div>
</div>
""", unsafe_allow_html=True)