import os
# ปิด torch.load แบบ weights-only (Torch ≥ 2.6 จะ default เป็น weights_only=True)
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
import io
import requests
import numpy as np
import streamlit as st
from PIL import Image
from dotenv import load_dotenv

# --------- ป้องกันปัญหา weights_only (Torch ≥ 2.6) ---------
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from ultralytics import YOLO

# --------- โหลด .env / Streamlit secrets ---------
load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH", st.secrets.get("MODEL_PATH", "best.pt"))
RAW_URL    = os.getenv("RAW_URL",    st.secrets.get("RAW_URL",    ""))

# --------- ฟังก์ชันดาวน์โหลดน้ำหนัก พร้อม progress bar ---------
def _download(url: str, dst: str):
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        prog = st.progress(0, text="📥 Downloading model…")
        done = 0
        with open(dst, "wb") as f:
            for chunk in r.iter_content(1 << 15):
                f.write(chunk)
                done += len(chunk)
                if total:
                    prog.progress(done / total, text=f"📥 {done/1e6:.1f}/{total/1e6:.1f} MB")
        prog.progress(1.0, text="✅ Download finished")

# --------- โหลดโมเดล (cache ตลอด session) ---------
@st.cache_resource(show_spinner="🚀 Loading model…")
def load_model():
    if not os.path.exists(MODEL_PATH):
        if not RAW_URL:
            st.error("❌ ไม่พบ MODEL_PATH และไม่ได้ตั้ง RAW_URL")
            st.stop()
        _download(RAW_URL, MODEL_PATH)
    return YOLO(MODEL_PATH)

model = load_model()

# --------- หน้า UI ---------
st.title("🕵️ Crack Detection with YOLOv11 + Streamlit")

uploaded = st.file_uploader("⬆️ Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Original Image", use_container_width=True)

    with st.spinner("Detecting…"):
        results = model.predict(
            source=np.asarray(img),
            imgsz=1088,
            conf=0.25,
            device="cpu"  # เปลี่ยนเป็น "0" ถ้ามี GPU
        )

    annotated = results[0].plot()
    st.image(annotated, caption="Detection Result", use_container_width=True)

    # ปุ่มดาวน์โหลดภาพผลลัพธ์
    buf = io.BytesIO()
    Image.fromarray(annotated).save(buf, format="PNG")
    st.download_button(
        "💾 Download Result",
        data=buf.getvalue(),
        file_name="detection_result.png",
        mime="image/png"
    )
