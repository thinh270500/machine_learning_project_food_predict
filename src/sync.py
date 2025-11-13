import os
import shutil
import gdown
from PIL import Image
import pillow_heif  # Cần: pip install pillow-heif

def download_from_drive(folder_id_or_url, local_path="dataset/raw"):
    os.makedirs(local_path, exist_ok=True)
    print(f"🔽 Đang tải dữ liệu từ Drive: {folder_id_or_url} ...")
    try:
        gdown.download_folder(
            id=None,
            url=folder_id_or_url,
            output=local_path,
            quiet=False,
            use_cookies=False
        )
        print(f"✅ Tải hoàn tất. Dữ liệu lưu tại: {local_path}")
    except Exception as e:
        print(f"❌ Lỗi khi tải từ Drive: {e}")

def convert_heic_to_jpg(root_dir):
    """
    Quét toàn bộ thư mục và chuyển file .heic thành .jpg
    """
    converted = 0
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".heic"):
                src_path = os.path.join(root, file)
                dst_path = os.path.splitext(src_path)[0] + ".jpg"
                try:
                    heif_file = pillow_heif.read_heif(src_path)
                    image = Image.frombytes(
                        heif_file.mode, heif_file.size, heif_file.data
                    )
                    image.save(dst_path, format="JPEG", quality=95)
                    os.remove(src_path)
                    converted += 1
                except Exception as e:
                    print(f"⚠️ Lỗi chuyển {file}: {e}")
    print(f"✅ Đã chuyển {converted} ảnh HEIC sang JPG")

def rename_images_in_folder(root_dir):
    """
    Đổi tên ảnh theo mẫu <ten_thu_muc><so_thu_tu>.<duoi_anh>
    """
    for class_name in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        images = [f for f in os.listdir(class_dir)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        images.sort()
        for idx, img_name in enumerate(images, start=1):
            ext = os.path.splitext(img_name)[1]
            new_name = f"{class_name.lower()}{idx}{ext}"
            src = os.path.join(class_dir, img_name)
            dst = os.path.join(class_dir, new_name)
            if src != dst and not os.path.exists(dst):
                shutil.move(src, dst)
        print(f"✅ Đã đổi tên {len(images)} ảnh trong: {class_name}")

if __name__ == "__main__":
    DRIVE_URL = "https://drive.google.com/drive/folders/1tDYQhEZy_WovYko2swNTZcbG8XAC68FQ?usp=sharing"
    LOCAL_PATH = "dataset/raw"

    # 1️⃣ Tải dữ liệu từ Drive
    download_from_drive(DRIVE_URL, LOCAL_PATH)

    # 2️⃣ Chuyển HEIC → JPG
    convert_heic_to_jpg(LOCAL_PATH)

    # 3️⃣ Đổi tên file ảnh
    rename_images_in_folder(LOCAL_PATH)

