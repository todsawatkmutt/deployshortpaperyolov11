import os
import io
import requests
import numpy as np
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from pickle import UnpicklingError

# ─── บังคับ weights_only=False ──────────────────────────────
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

from ultralytics import YOLO

# ─── โหลด .env / secrets ────────────────────────────────────
load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH", st.secrets.get("MODEL_PATH", "best.pt"))
RAW_URL    = os.getenv("RAW_URL",    st.secrets.get("RAW_URL",    ""))

def _download(url: str, dst: str):
    """ดาวน์โหลด .pt พร้อม progress bar"""
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
                    prog.progress(done / total,
                                 text=f"📥 {done/1e6:.1f}/{total/1e6:.1f} MB")
        prog.progress(1.0, text="✅ Download finished")

@st.cache_resource(show_spinner="🚀 Loading model…")
def load_model():
    # ถ้าไฟล์ไม่มี ให้ดาวน์โหลด
    if not os.path.exists(MODEL_PATH):
        if not RAW_URL:
            st.error("❌ ไม่พบ MODEL_PATH และ RAW_URL")
            st.stop()
        _download(RAW_URL, MODEL_PATH)

    # พยายามโหลดโมเดล ถ้าไฟล์เสีย (UnpicklingError) ให้ลบแล้วดาวน์โหลดใหม่
    try:
        return YOLO(MODEL_PATH)
    except UnpicklingError:
        st.warning("⚠️ Model file corrupted, re-downloading…")
        os.remove(MODEL_PATH)
        _download(RAW_URL, MODEL_PATH)
        return YOLO(MODEL_PATH)

model = load_model()

# ─── UI ──────────────────────────────────────────────────────
st.title("🕵️ Crack Detection with YOLOv11")

uploaded = st.file_uploader("⬆️ Upload an image (JPG/PNG)", type=["jpg","jpeg","png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Original", use_container_width=True)

    with st.spinner("Detecting…"):
        results = model.predict(
            source=np.asarray(img),
            imgsz=1088, conf=0.25, device="cpu"
        )
    annotated = results[0].plot()
    st.image(annotated, caption="Detection Result", use_container_width=True)

    buf = io.BytesIO()
    Image.fromarray(annotated).save(buf, format="PNG")
    st.download_button("💾 Download result", buf.getvalue(),
                       file_name="result.png", mime="image/png")
