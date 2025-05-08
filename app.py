import os, io, tempfile, requests
import numpy as np
import streamlit as st
from PIL import Image

# --------- ป้องกันปัญหา weights_only (Torch ≥ 2.6) ---------
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from ultralytics import YOLO

# --------- โหลด .env / secrets ---------
from dotenv import load_dotenv
load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", st.secrets.get("MODEL_PATH", "best.pt"))
RAW_URL    = os.getenv("RAW_URL",    st.secrets.get("RAW_URL", ""))

# --------- ฟังก์ชันดาวน์โหลดน้ำหนัก ---------
def _download(url: str, dst: str):
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        tot = int(r.headers.get("content-length", 0))
        with st.progress(0, text="📥 Downloading model…") as bar, open(dst, "wb") as f:
            sz = 0
            for chunk in r.iter_content(1 << 15):
                f.write(chunk)
                sz += len(chunk)
                bar.progress(min(sz / tot, 1.0) if tot else 0.0)

# --------- โหลดโมเดล ครั้งเดียวต่อเซสชัน ---------
@st.cache_resource(show_spinner="🚀 Loading model…")
def load_model():
    if not os.path.exists(MODEL_PATH):
        if not RAW_URL:
            st.error("❌ ไม่พบ MODEL_PATH และไม่ได้ตั้ง RAW_URL")
            st.stop()
        _download(RAW_URL, MODEL_PATH)
    return YOLO(MODEL_PATH)

model = load_model()

# --------- UI ---------
st.title("🕵️ Crack Detection (YOLOv11)")

uploaded = st.file_uploader("⬆️ Upload image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded:
    # Streamlit ให้ไฟล์เป็น BytesIO -> PIL -> NumPy
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Original", use_container_width=True)

    # ---------- Predict ----------
    with st.spinner("Detecting…"):
        results = model.predict(
            source=np.asarray(img),  # direct NumPy
            imgsz=1088,
            conf=0.25,
            device="cpu"            # > เปลี่ยนเป็น 0 ถ้ามี GPU
        )
    annotated = results[0].plot()

    st.image(annotated, caption="Detection Result", use_container_width=True)

    # ดาวน์โหลดผลลัพธ์ (optional)
    buf = io.BytesIO()
    Image.fromarray(annotated).save(buf, format="PNG")
    st.download_button("💾 Download result", buf.getvalue(), "result.png", "image/png")
