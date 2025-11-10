import os
import io
from rembg import remove
from PIL import Image


def remove_background_from_image(input_path, output_path, bg_color=(255, 255, 255)):
    """
    Tách nền 1 ảnh bằng thư viện rembg (AI segment tự động)
    và ghép nền trắng (hoặc màu tùy chọn) để tránh viền mờ.
    """
    # Đọc dữ liệu gốc
    with open(input_path, "rb") as inp_file:
        input_data = inp_file.read()

    # Tách nền (rembg trả về ảnh PNG có alpha)
    result = remove(input_data)

    # Mở bằng PIL
    img = Image.open(io.BytesIO(result)).convert("RGBA")

    # Tạo nền trắng (RGB) cùng kích thước
    bg = Image.new("RGB", img.size, bg_color)

    # Dán ảnh RGBA lên nền dùng alpha làm mask
    bg.paste(img, mask=img.split()[3])

    # Lưu file (đảm bảo không còn kênh alpha)
    bg.save(output_path, format="PNG")


def remove_background_dataset(input_dir="Dataset/Raw", output_dir="Dataset/Cleaned", bg_color=(255, 255, 255)):
    """
    Duyệt toàn bộ thư mục dataset/raw và tách nền toàn bộ ảnh,
    sau đó ghép nền trắng, lưu sang dataset/cleaned giữ nguyên cấu trúc thư mục.
    """
    os.makedirs(output_dir, exist_ok=True)

    for class_name in os.listdir(input_dir):
        class_input_dir = os.path.join(input_dir, class_name)
        class_output_dir = os.path.join(output_dir, class_name)

        if not os.path.isdir(class_input_dir):
            continue

        os.makedirs(class_output_dir, exist_ok=True)

        images = [
            f for f in os.listdir(class_input_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        print(f"\n🟢 Đang xử lý lớp: {class_name} ({len(images)} ảnh)")

        for img_name in images:
            input_path = os.path.join(class_input_dir, img_name)
            output_path = os.path.join(class_output_dir, img_name.replace(".jpg", ".png"))

            try:
                remove_background_from_image(input_path, output_path, bg_color)
                print(f"  ✅ {img_name} → {output_path}")
            except Exception as e:
                print(f"  ⚠️ Lỗi với {img_name}: {e}")

    print("\n🎉 Hoàn tất tách nền! Dữ liệu lưu tại:", output_dir)


if __name__ == "__main__":
    remove_background_dataset(
        input_dir="Dataset/Raw",
        output_dir="Dataset/Cleaned",
        bg_color=(255, 255, 255)  # có thể đổi thành (0,128,0) nếu muốn nền xanh lá nhạt
    )
