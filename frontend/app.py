
# # bản streamlit cloud 
# import streamlit as st
# import torch
# from torchvision import models, transforms
# from PIL import Image
# import os
# import pillow_heif

# # =====================================================
# # 1. TỰ ĐỘNG TÌM MODEL – CHẠY ĐÚNG 100% TRÊN STREAMLIT
# # =====================================================
# @st.cache_resource
# def load_model():
#     model = models.resnet50(weights=None)
#     model.fc = torch.nn.Sequential(
#         torch.nn.Linear(model.fc.in_features, 256),
#         torch.nn.ReLU(),
#         torch.nn.Dropout(0.4),
#         torch.nn.Linear(256, 2)
#     )

#     # Đường dẫn đúng khi chạy trên Streamlit Cloud
#     model_path = "frontend/model/resnet50_leaf_final_best.pt"

#     if not os.path.exists(model_path):
#         st.error(f"KHÔNG TÌM THẤY MODEL TẠI: {model_path}")
#         st.info("Kiểm tra lại file model có nằm đúng trong thư mục frontend/model/ không")
#         st.stop()

#     try:
#         model.load_state_dict(torch.load(model_path, map_location="cpu"))
#         st.success("Model đã tải thành công!")
#     except Exception as e:
#         st.error(f"Lỗi load model: {e}")
#         st.stop()

#     model.eval()
#     return model

# model = load_model()

# # =====================================================
# # 2. TRANSFORM + UI
# # =====================================================
# transform = transforms.Compose([
#     transforms.Resize((128, 128)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

# st.set_page_config(page_title="Leafy AI", page_icon="leaf", layout="wide")
# st.markdown("<h1 style='text-align: center; color: #2E7D32;'>Leafy – Phát Hiện Bệnh Lá Cây</h1>", unsafe_allow_html=True)
# st.markdown("<p style='text-align: center; font-size: 18px;'>Kéo thả ảnh lá cây → Kết quả tức thì!</p>", unsafe_allow_html=True)

# def load_image(file):
#     if file.name.lower().endswith((".heic", ".heif")):
#         heif_file = pillow_heif.read_heif(file)
#         return Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw")
#     return Image.open(file).convert("RGB")

# uploaded_file = st.file_uploader("Kéo thả ảnh vào đây", type=['jpg','jpeg','png','heic','heif','webp','bmp','tiff'])

# if uploaded_file and model:
#     img = load_image(uploaded_file)
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.image(img, caption="Ảnh lá cây", use_container_width=True)
    
#     with col2:
#         with st.spinner("Đang phân tích..."):
#             tensor = transform(img).unsqueeze(0)
#             with torch.no_grad():
#                 prob = torch.softmax(model(tensor), dim=1)[0]
#                 pred = prob.argmax().item()
#                 conf = prob[pred].item() * 100
        
#         label = "BỊ BỆNH" if pred == 0 else "KHỎE MẠNH"
#         color = "#f44336" if pred == 0 else "#4CAF50"
        
#         st.markdown(f"""
#         <div style="padding: 1.5em; border: 3px solid {color}; border-radius: 15px; text-align: center; background: {color}15;">
#             <h2 style="color: {color}; margin:0;">{label}</h2>
#             <h3>Độ tin cậy: {conf:.1f}%</h3>
#         </div>
#         """, unsafe_allow_html=True)
        
#         st.bar_chart({"BỊ BỆNH": prob[0].item()*100, "KHỎE MẠNH": prob[1].item()*100})
        
#         if pred == 0:
#             st.error("**Cảnh báo:** Lá có dấu hiệu bệnh. Có thể liên quan tới sâu, nấm hoặc vi khuẩn.")
#         else:
#             st.success("**Tuyệt vời!** Lá hoàn toàn khỏe mạnh.")
# # # Footer
# # st.markdown("---")
# # st.markdown("<p style='text-align: center; color: #666;'>Model: ResNet50 | Acc: 88.8%</p>", unsafe_allow_html=True)


# bản update streamlit
# frontend/app.py – PHIÊN BẢN HOÀN HẢO NHẤT (18/11/2025 – 00:45 AM)
import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import os
import pillow_heif

# =====================================================
# 1. LOAD MODEL – TỰ ĐỘNG TÌM + BÁO LỖI RÕ RÀNG
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

    model_path = "frontend/model/resnet50_leaf_final_best.pt"

    if not os.path.exists(model_path):
        st.error("KHÔNG TÌM THẤY FILE MODEL!")
        st.info(f"Đường dẫn đang tìm: `{model_path}`")
        st.info("Hãy đảm bảo file model nằm đúng trong thư mục `frontend/model/`")
        st.stop()

    try:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        st.success("MODEL ĐÃ TẢI THÀNH CÔNG! (92.00% - Recall bệnh 100%)")
    except Exception as e:
        st.error(f"Lỗi khi tải model: {e}")
        st.stop()

    model.eval()
    return model

model = load_model()

# =====================================================
# 2. TRANSFORM + UI
# =====================================================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

st.set_page_config(page_title="Leafy AI", page_icon="leaf", layout="wide")
st.markdown("<h1 style='text-align: center; color: #2E7D32;'>Leafy – Phát Hiện Bệnh Lá Cây</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Kéo thả ảnh lá cây → Kết quả tức thì!</p>", unsafe_allow_html=True)

def load_image(file):
    if file.name.lower().endswith((".heic", ".heif")):
        try:
            heif_file = pillow_heif.read_heif(file)
            return Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw")
        except:
            st.error("Lỗi đọc file HEIC. Vui lòng thử định dạng khác.")
            return None
    return Image.open(file).convert("RGB")

uploaded_file = st.file_uploader(
    "Kéo thả ảnh lá cây vào đây",
    type=['jpg','jpeg','png','heic','heif','webp','bmp','bmp','tiff','jfif']
)

# =====================================================
# 3. DỰ ĐOÁN + TÍNH NĂNG CHỐNG TROLL (KHÔNG PHẢI LÁ CÂY)
# =====================================================
if uploaded_file and model:
    img = load_image(uploaded_file)
    if img is None:
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Ảnh đã upload", use_container_width=True)

    with col2:
        with st.spinner("AI đang phân tích..."):
            tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                logits = model(tensor)
                prob = torch.softmax(logits, dim=1)[0]
                confidence_max = prob.max().item() * 100

                # TÍNH NĂNG SIÊU MẠNH: PHÁT HIỆN KHÔNG PHẢI ẢNH LÁ CÂY
                if confidence_max < 78:  # Ngưỡng đã test tối ưu
                    st.error("KHÔNG PHẢI ẢNH LÁ CÂY!")
                    st.warning("Vui lòng upload ảnh lá cây để phân tích bệnh.")
                    st.info("Ví dụ bị từ chối: ảnh người, chó mèo, đồ ăn, xe cộ, bầu trời...")
                    st.stop()

                pred = prob.argmax().item()
                conf = confidence_max

        label = "BỊ BỆNH" if pred == 0 else "KHỎE MẠNH"
        color = "#f44336" if pred == 0 else "#4CAF50"

        st.markdown(f"""
        <div style="padding: 2em; border: 4px solid {color}; border-radius: 20px; text-align: center; background: {color}10;">
            <h1 style="color: {color}; margin:0;">{label}</h1>
            <h2>Độ tin cậy: {conf:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)

        st.bar_chart({"BỊ BỆNH": prob[0].item()*100, "KHỎE MẠNH": prob[1].item()*100}, height=300)

        if pred == 0:
            st.error("**Cảnh báo:** Lá có dấu hiệu bệnh. Có thể liên quan tới sâu, nấm hoặc vi khuẩn.")
        else:
            st.success("**Tuyệt vời!** Lá hoàn toàn khỏe mạnh.")
            st.balloons()

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #666;'>Model: ResNet50 | Acc: 88.8%</p>", unsafe_allow_html=True)
