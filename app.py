from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps

# กำหนดเส้นทางโฟลเดอร์และไฟล์
HERE = Path(__file__).parent
MODEL_PATH = HERE / "model" / "model.pkl"  # แก้ไขเป็นไฟล์ .pkl แล้ว
LABELS_PATH = HERE / "model" / "labels.json"
TOP_K = 5

# ค่าสถิติ ImageNet (ใช้ในกรณีที่โมเดลเทรนมาแบบ Normalize)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ตั้งค่าหน้าเว็บ Streamlit
st.set_page_config(page_title="Image Classifier", page_icon="🖼️", layout="centered")


@st.cache_resource(show_spinner="กำลังโหลดโมเดล…")
def load_model():
    # ใช้ torch.load สำหรับไฟล์ .pkl ของ PyTorch
    # map_location="cpu" เพื่อให้รันบนเครื่องที่ไม่มีการ์ดจอได้
    model = torch.load(str(MODEL_PATH), map_location="cpu")
    model.eval()  # ตั้งค่าให้อยู่ในโหมดประเมินผล
    
    # อ่านไฟล์ labels.json (เพิ่ม encoding="utf-8" เพื่อรองรับภาษาไทย)
    meta = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    return model, meta["labels"], meta["preprocess"]


def preprocess(pil_img: Image.Image, pre: dict) -> torch.Tensor:
    """จำลองขั้นตอนการทำ Preprocess ก่อนนำเข้าโมเดล"""
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


# --- ส่วนของหน้าจอการแสดงผล (UI) ---
try:
    model, labels, pre_config = load_model()
except Exception as e:
    st.error(f"ไม่สามารถโหลดโมเดลได้ โปรดตรวจสอบว่ามีไฟล์ `model.pkl` และ `labels.json` ในโฟลเดอร์ `model` แล้วหรือยัง\n\n**รายละเอียด:** {e}")
    st.stop()

st.title("🖼️ แอปพลิเคชันจำแนกรูปภาพด้วย AI")
st.write("อัปโหลดรูปภาพของคุณเพื่อให้โมเดลช่วยวิเคราะห์ผลลัพธ์")

uploaded_file = st.file_uploader("เลือกรูปภาพของคุณ (รองรับ JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # เปิดรูปภาพขึ้นมาและจัดการเรื่องการกลับหัว (Exif)
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image) 
    
    # แสดงรูปภาพ
    st.image(image, caption="รูปภาพที่อัปโหลดสำเร็จ", use_container_width=True)
    
    with st.spinner("โมเดลกำลังประมวลผลรูปภาพ..."):
        results = classify(model, labels, pre_config, image)
        
    st.subheader("📊 ผลการวิเคราะห์ (Top 5)")
    for label, prob in results[:TOP_K]:
        percent = prob * 100
        st.write(f"**{label}**: {percent:.2f}%")
        st.progress(prob)