from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np
import os
from PIL import Image, ImageEnhance, ImageFilter
import random

# ---- Thêm nền trắng nếu ảnh có alpha channel ----
def handle_alpha_channel(image, background_color=(255, 255, 255)):
    if image.mode in ('RGBA', 'LA'):
        background = Image.new("RGB", image.size, background_color)
        background.paste(image, mask=image.split()[-1])
        return background
    return image.convert("RGB")

# ---- Thêm viền trắng để tránh kéo mép khi augment ----
def add_white_border(img, border_size=30):
    w, h = img.size
    new_img = Image.new("RGB", (w + 2 * border_size, h + 2 * border_size), (255, 255, 255))
    new_img.paste(img, (border_size, border_size))
    return new_img

# ---- Cắt ảnh vuông và resize ----
def center_crop_square(image, size=256):
    w, h = image.size
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    image = image.crop((left, top, left + min_dim, top + min_dim))
    return image.resize((size, size))

# ---- Augment thủ công: thêm noise, blur, màu sắc ----
def random_effects(img):
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

    if random.random() < 0.3:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.7, 1.3))

    if random.random() < 0.3:
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(random.uniform(0.6, 1.4))

    if random.random() < 0.3:
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(random.uniform(0.5, 2.0))

    if random.random() < 0.3:
        img = img.point(lambda p: p + random.randint(-15, 15))  # lệch sáng nhẹ

    return img

# ---- Augment cho một ảnh ----
def augment_image(img_path, save_dir, class_name, base_name, n_samples=6):
    os.makedirs(save_dir, exist_ok=True)
    img = load_img(img_path)
    img = handle_alpha_channel(img)
    img = add_white_border(img, 30)
    img = center_crop_square(img)

    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)

    datagen = ImageDataGenerator(
        rotation_range=12,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        shear_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,        # bật flip dọc
        brightness_range=[0.8, 1.3],
        fill_mode='constant',
        cval=255
    )

    for i, batch in enumerate(datagen.flow(x, batch_size=1)):
        new_img = Image.fromarray(batch[0].astype('uint8'))
        new_img = random_effects(new_img)   # thêm biến đổi màu/ngẫu nhiên

        new_name = f"{class_name}_{base_name}_aug{i+1}.png"
        save_path = os.path.join(save_dir, new_name)
        new_img.save(save_path)

        if i >= n_samples - 1:
            break

# ---- Augment toàn bộ dataset ----
def augment_dataset(raw_dir="dataset/cleaned", augmented_dir="dataset/augmented", n_samples_per_img=6):
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
            base_name = os.path.splitext(img_name)[0]
            augment_image(img_path, dest_dir, cls, base_name, n_samples=n_samples_per_img)
            print(f"  ✅ Augmented: {img_name}")

    print(f"\n🎉 Augmentation completed! Results saved in: {augmented_dir}")

# ---- Chạy thử ----
if __name__ == "__main__":
    augment_dataset(
        raw_dir="dataset/cleaned",
        augmented_dir="dataset/augmented",
        n_samples_per_img=8   # tăng số lượng để có thêm biến thể
    )
