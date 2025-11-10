import os
import shutil
import random
from tqdm import tqdm
import math

# ====== Cấu hình ======
base_dir = r"D:\CaoHoc\May Hoc\food ingredient ai\Dataset"
cleaned_dir = os.path.join(base_dir, "Cleaned")
augmented_dir = os.path.join(base_dir, "Augmented")
output_dir = base_dir # tạo train/val/test ngay trong Dataset

classes = ["Healthy", "Diseases"]
splits = ["Train", "Validation", "Test"]

# --- Tỷ lệ mới ---
# Tỷ lệ này sẽ áp dụng cho tập ảnh GỐC (cleaned)
split_ratio = {
"Train_Cleaned": 0.75, # lấy từ augmented
"Validation": 0.15, # lấy ảnh từ raw (ảnh gốc)
"Test": 0.10 # lấy ảnh tù raw
}

# ====== Hàm hỗ trợ ======
def create_dirs():
# Xóa thư mục cũ trước để đảm bảo dữ liệu mới được tạo
    for split in splits:
        dir_path = os.path.join(output_dir, split)
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    # Tạo thư mục mới
    for split in splits:
        for cls in classes:
            os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

def copy_images(src_list, dest_dir):
# Dùng tqdm để hiển thị tiến trình khi copy
    for img in tqdm(src_list, desc=f"Copying to {os.path.basename(dest_dir)}"):
        shutil.copy(img, dest_dir)

# ====== Chia dữ liệu ======
create_dirs()

for cls in classes:
    print(f"\n📂 Xử lý lớp: {cls}")

    # Ảnh cleaned
    cleaned_path = os.path.join(cleaned_dir, cls)
    cleaned_images = [
        os.path.join(cleaned_path, f)
        for f in os.listdir(cleaned_path)
        if f.lower().endswith(('.jpg', '.png', '.jpeg'))
    ]

    # Ảnh augmented (dùng để thêm vào tập train)
    augmented_path = os.path.join(augmented_dir, cls)
    augmented_images = [
        os.path.join(augmented_path, f)
        for f in os.listdir(augmented_path)
        if f.lower().endswith(('.jpg', '.png', '.jpeg'))
    ]

    # --- Shuffle & chia ---
    random.shuffle(cleaned_images)
    total_cleaned = len(cleaned_images)

    test_size = math.ceil(total_cleaned * split_ratio["Test"])
    val_size = math.ceil(total_cleaned * split_ratio["Validation"])

    # Đảm bảo tổng Val + Test không vượt quá tổng Cleaned
    if val_size + test_size >= total_cleaned:
        # Nếu tổng Val/Test quá lớn, đặt Val/Test tối thiểu là 1 và chia đều phần còn lại
        test_size = max(1, math.floor(total_cleaned * split_ratio["Test"]))
        val_size = max(1, total_cleaned - test_size)

    val_images = cleaned_images[:val_size]
    test_images = cleaned_images[val_size:val_size + test_size]
    train_cleaned = cleaned_images[val_size + test_size:]

    # Train = cleaned còn lại + augmented
    train_images = train_cleaned + augmented_images
    random.shuffle(train_images)

    # --- Thông tin thống kê ---
    print(f"  ➤ Cleaned total: {total_cleaned} | Augmented: {len(augmented_images)}")
    print(f"  ➤ Chia Val/Test từ Cleaned: Val={len(val_images)} ({len(val_images)/total_cleaned:.2%}) | Test={len(test_images)} ({len(test_images)/total_cleaned:.2%})")
    print(f"  ➤ Train CUỐI CÙNG: {len(train_images)} ảnh")

    # --- Sao chép ---
    copy_images(train_images, os.path.join(output_dir, "Train", cls))
    copy_images(val_images, os.path.join(output_dir, "Validation", cls))
    copy_images(test_images, os.path.join(output_dir, "Test", cls))

print("\n✅ Hoàn tất chia dataset. Hãy chạy code huấn luyện với Data Augmentation và các tham số mô hình mới.")