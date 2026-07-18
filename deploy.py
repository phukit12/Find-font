from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
import pickle

import streamlit as st
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

HERE = Path(__file__).parent
TOP_K = 5

st.set_page_config(page_title="Thai Font Finder", page_icon="🔍", layout="wide")

# --- ฟังก์ชันแปลงรูปเป็น Base64 เพื่อให้ใส่ใน HTML ได้ ---
def image_to_base64(img: Image.Image):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# --- CSS & Header HTML ---
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Sarabun:wght@400;500;600&display=swap" rel="stylesheet">
<style>
.stApp { background-color: #f0ead8; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
.tff-page { background: transparent; min-height: 100vh; font-family: 'Sarabun', sans-serif; }
.tff-header { position: relative; background: #f0ead8; padding: 1.6rem 1rem 1.3rem; text-align: center; overflow: hidden; min-height: 130px; display: flex; flex-direction: column; align-items: center; justify-content: center; border-bottom: 0.5px solid #d6cebc; }
.tff-hbg { position: absolute; inset: 0; overflow: hidden; display: flex; flex-direction: column; justify-content: center; pointer-events: none; }
.tff-hbg-row { white-space: nowrap; font-size: 30px; font-family: 'Sarabun', sans-serif; color: rgba(80,60,20,0.13); line-height: 1.6; letter-spacing: 8px; }
.tff-logo { position: relative; z-index: 2; display: flex; align-items: center; justify-content: center; gap: 8px; font-family: 'Sarabun', sans-serif; }
.tff-logo-kor { font-size: 48px; color: #2d5a3d; font-weight: 600; line-height: 1; }
.tff-logo-text { font-size: 28px; font-weight: 600; color: #2c2c2a; letter-spacing: 1px; }
.tff-logo-a { font-size: 38px; font-weight: 600; color: #2c2c2a; }
.tff-subtitle { position: relative; z-index: 2; margin-top: 8px; background: #d6cebc; color: #4a4640; font-size: 13px; padding: 3px 18px; border-radius: 20px; display: inline-block; font-family: 'Sarabun', sans-serif; }
.tff-results-title { font-size: 17px; font-weight: 600; color: #2c2c2a; margin-bottom: 12px; font-family: 'Sarabun', sans-serif; }
.tff-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; margin-bottom: 8px; }
.tff-card { background: rgba(255,255,255,0.55); border: 0.5px solid #d3d1c7; border-radius: 10px; padding: 10px 12px; font-family: 'Sarabun', sans-serif; }
.tff-card.top { background: rgba(200,230,210,0.65); border: 1px solid #5a9a6a; }
.tff-card-top { display: flex; align-items: flex-start; justify-content: space-between; margin-bottom: 2px; }
.tff-rank { font-size: 18px; font-weight: 500; color: #888780; }
.tff-name { font-size: 13px; font-weight: 500; color: #2c2c2a; margin-bottom: 4px; }
.tff-pct-high { font-size: 14px; font-weight: 600; color: #2d5a3d; }
.tff-pct-mid  { font-size: 13px; font-weight: 600; color: #b07d10; }
.tff-pct-low  { font-size: 13px; color: #888780; }
.tff-bar-bg   { height: 3px; background: #d3d1c7; border-radius: 2px; margin-top: 4px; }
.tff-bar      { height: 3px; border-radius: 2px; }
.tff-bar-green  { background: #3b7a50; }
.tff-bar-yellow { background: #d4a017; }
.tff-bar-gray   { background: #b4b2a9; }
.tff-preview-box { border: 1.5px dashed #b4b2a9; border-radius: 10px; background: rgba(255,255,255,0.35); padding: 1rem; min-height: 220px; display: flex; flex-direction: column; align-items: center; justify-content: center; font-family: 'Sarabun', sans-serif; color: #888780; font-size: 13px; overflow: hidden; }
</style>

<div class="tff-page">
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
</div>
""", unsafe_allow_html=True)

# --- ระบบโหลดโครงสร้างโมเดลและประกบไฟล์แยกส่วน (.pth) ---
@st.cache_resource(show_spinner="กำลังประกอบร่างโมเดล...")
def load_split_model():
    # 1. ดึงรายชื่อคลาสฟอนต์
    with open(HERE / "model/font_classes.pkl", "rb") as f:
        labels = pickle.load(f)
        
    # 2. จำลองโครงสร้าง ResNet34 ตามสไตล์ FastAI
    model = models.resnet34(weights=None)
    num_features = model.fc.in_features
    
    # เปลี่ยนหัว Head ให้ตรงกับตอนที่เทรนใน FastAI
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.5),
        nn.Linear(512, len(labels))
    )
    
    # 3. โหลดเฉพาะ Weights ส่วนตัวเครื่อง (weight_body) ที่มีขนาดไม่เกินลิมิตขึ้นมาประกอบ
    state_dict = torch.load(HERE / "model/weight_body.pth", map_location="cpu", weights_only=False)
    
    # กรองคีย์ส่วนหัวออกในกรณีประกบโครงสร้างมาตรฐาน
    model_dict = model.state_dict()
    weights_to_load = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(weights_to_load)
    model.load_state_dict(model_dict)
    
    model.eval()
    return model, labels

try:
    model, labels = load_split_model()
except Exception as e:
    st.error(f"ไม่สามารถประกอบร่างโมเดลได้: {e}")
    st.info("คำแนะนำ: ตรวจสอบว่ามีโฟลเดอร์ชื่อ 'model' ที่บรรจุไฟล์ 'weight_body.pth' และ 'font_classes.pkl' อยู่ในโปรเจกต์แล้ว")
    st.stop()

# --- ตัวแปรรูปภาพสำหรับโมเดล PyTorch ---
img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

col_left, col_right = st.columns([1, 1.4])

with col_left:
    st.markdown('<div style="padding: 1rem 1rem 0.5rem; font-family: Sarabun, sans-serif;">', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("นำเข้ารูปภาพ", type=["jpg", "jpeg", "png", "webp", "bmp"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        image = ImageOps.exif_transpose(image)
        
        img_b64 = image_to_base64(image)
        preview_html = f'''
        <div class="tff-preview-box" style="padding: 0;">
            <img src="data:image/png;base64,{img_b64}" style="width: 100%; height: 100%; object-fit: contain; border-radius: 8px;">
        </div>
        '''
        st.markdown(preview_html, unsafe_allow_html=True)
    else:
        st.markdown('<div class="tff-preview-box">พรีวิวรูปภาพที่อัปโหลด</div>', unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div style="padding: 1rem 1rem 0;">', unsafe_allow_html=True)
    st.markdown('<div class="tff-results-title">ผลการพยากรณ์</div>', unsafe_allow_html=True)

    if uploaded_file:
        with st.spinner("กำลังวิเคราะห์..."):
            # สั่งพยากรณ์ผลผ่าน PyTorch โดยตรง
            input_tensor = img_transforms(image).unsqueeze(0)
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1)[0]
            
            results = sorted(
                ((labels[i], float(probs[i])) for i in range(len(labels))),
                key=lambda p: p[1],
                reverse=True,
            )

        def pct_class(p):
            if p >= 0.85: return "tff-pct-high"
            if p >= 0.60: return "tff-pct-mid"
            return "tff-pct-low"

        def bar_class(p):
            if p >= 0.85: return "tff-bar-green"
            if p >= 0.60: return "tff-bar-yellow"
            return "tff-bar-gray"

        top5 = results[:TOP_K]
        
        # --- แถวที่ 1 (อันดับ 1-3) ---
        cards_html = '<div class="tff-grid">'
        for i, (label, prob) in enumerate(top5[:3]):
            top_cls = "top" if i == 0 else ""
            pct_str = f"{prob*100:.1f}%"
            bar_w = int(prob * 100)
            cards_html += f"""
            <div class="tff-card {top_cls}">
                <div class="tff-card-top">
                    <span class="tff-rank">{i+1}</span>
                    <span class="{pct_class(prob)}">{pct_str}</span>
                </div>
                <div class="tff-name">{label}</div>
                <div class="tff-bar-bg"><div class="tff-bar {bar_class(prob)}" style="width:{bar_w}%"></div></div>
            </div>"""
        cards_html += '</div>'

        # --- แถวที่ 2 (อันดับ 4-5) ---
        cards_html += '<div class="tff-grid">'
        for i, (label, prob) in enumerate(top5[3:]):
            pct_str = f"{prob*100:.1f}%"
            bar_w = int(prob * 100)
            cards_html += f"""
            <div class="tff-card">
                <div class="tff-card-top">
                    <span class="tff-rank">{i+4}</span>
                    <span class="{pct_class(prob)}">{pct_str}</span>
                </div>
                <div class="tff-name">{label}</div>
                <div class="tff-bar-bg"><div class="tff-bar {bar_class(prob)}" style="width:{bar_w}%"></div></div>
            </div>"""
        
        cards_html += '<div class="tff-card" style="background:transparent;border:none;"></div>'
        cards_html += '</div>'

        st.markdown(cards_html, unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:#888780;font-size:14px;font-family:Sarabun,sans-serif;">อัปโหลดรูปภาพเพื่อดูผลการพยากรณ์</p>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)