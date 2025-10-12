from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

# ---- Hàm crop vuông ----
def center_crop_square(image, size=256):
    w, h = image.size
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    image = image.crop((left, top, left + min_dim, top + min_dim))
    return image.resize((size, size))

# ---- Hàm augment một ảnh ----
def augment_image(img_path, save_dir, n_samples=5):
    os.makedirs(save_dir, exist_ok=True)
    img = load_img(img_path)
    img = center_crop_square(img)

    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)

    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.9, 1.1]
    )

    for i, batch in enumerate(datagen.flow(
        x, batch_size=1, save_to_dir=save_dir, save_prefix='aug', save_format='png'
    )):
        if i >= n_samples:
            break

# ---- Hàm augment toàn bộ dataset ----
def augment_dataset(raw_dir="dataset/raw", augmented_dir="dataset/augmented", n_samples_per_img=5):
    classes = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    print(f"🔍 Found classes: {classes}")

    for cls in classes:
        src_dir = os.path.join(raw_dir, cls)
        dest_dir = os.path.join(augmented_dir, cls)
        os.makedirs(dest_dir, exist_ok=True)

        images = [f for f in os.listdir(src_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        print(f"\n📂 Processing class: {cls} ({len(images)} images)")

        for img_name in images:
            img_path = os.path.join(src_dir, img_name)
            augment_image(img_path, dest_dir, n_samples=n_samples_per_img)
            print(f"  ✅ Augmented: {img_name}")

    print("\n🎉 Augmentation completed! All results saved in abc:", augmented_dir)

# ---- Chạy thử ----
if __name__ == "__main__":
    augment_dataset(
        raw_dir="dataset/raw",
        augmented_dir="dataset/augmented",
        n_samples_per_img=6 # số lượng ảnh sinh thêm mỗi ảnh gốc
    )
