# app.py
import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np

# =====================================================
# 1. LOAD MODEL
# =====================================================
@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(model.fc.in_features, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.4),
        torch.nn.Linear(256, 2)
    )
    try:
        model.load_state_dict(torch.load("model/resnet50_leaf_disease_final.pt", map_location="cpu"))
        st.success("Model đã tải thành công!")
    except Exception as e:
        st.error(f"Lỗi tải model: {e}")
        return None
    model.eval()
    return model

model = load_model()

# =====================================================
# 2. TRANSFORM
# =====================================================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# =====================================================
# 3. UI – 2 CỘT NGANG
# =====================================================
st.set_page_config(page_title="AI Lá Cây", page_icon="leaf", layout="wide") 

st.markdown("""
<style>
    .main {background-color: #f8fff8;}
    .stButton>button {
        background: linear-gradient(90deg, #4CAF50, #45a049);
        color: white;
        border-radius: 12px;
        height: 3em;
        font-weight: bold;
    }
    .result-box {
        padding: 1.5em;
        border-radius: 12px;
        text-align: center;
        margin: 1em 0;
        border: 2px solid;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #2E7D32;'>Leafy Phát Hiện Bệnh Lá Cây</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Upload ảnh → Nhận kết quả tức thì!</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")

if uploaded_file is not None and model is not None:
    img = Image.open(uploaded_file).convert("RGB")

    # CHIA 2 CỘT NGANG
    col_left, col_right = st.columns(2)

    with col_left:
        st.image(img, caption="Ảnh lá cây", use_container_width=True)

    with col_right:
        with st.spinner("Đang phân tích..."):
            img_tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                output = model(img_tensor)
                prob = torch.softmax(output, 1)[0]
                pred = torch.argmax(prob).item()
                conf = prob[pred].item() * 100

        label = "KHỎE MẠNH" if pred == 0 else "BỊ BỆNH"
        color = "#4CAF50" if pred == 0 else "#f44336"
        border_color = "#4CAF50" if pred == 0 else "#f44336"

        st.markdown(f"""
        <div class="result-box" style="border-color: {border_color};">
            <h2 style="color: {color}; margin:0;">{label}</h2>
            <h3>Độ tin cậy: {conf:.1f}%</h3>
        </div>
        """, unsafe_allow_html=True)

        # Biểu đồ cột
        chart_data = {"Khỏe mạnh": prob[0].item()*100, "Bị bệnh": prob[1].item()*100}
        st.bar_chart(chart_data)

        # Lời khuyên
        if pred == 1:
            st.error("**Cảnh báo:** Lá có dấu hiệu bệnh. Kiểm tra sâu, nấm, thiếu nước.")
        else:
            st.success("**Tuyệt vời!** Lá hoàn toàn khỏe mạnh.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #666;'>Model: ResNet50 | Acc: 85.79% |</p>", unsafe_allow_html=True)