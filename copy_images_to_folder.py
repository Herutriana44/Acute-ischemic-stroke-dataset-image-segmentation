import os
import shutil
import uuid

def is_image_file(filepath):
    """Deteksi file gambar berdasarkan ekstensi umum."""
    image_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif', '.webp'}
    return os.path.splitext(filepath)[1].lower() in image_extensions

def copy_and_rename_images(source_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, dirs, files in os.walk(source_folder):
        for file in files:
            source_path = os.path.join(root, file)

            if is_image_file(source_path):
                # Pertahankan ekstensi asli
                ext = os.path.splitext(file)[1].lower()
                new_filename = str(uuid.uuid4()) + ext
                destination_path = os.path.join(output_folder, new_filename)

                shutil.copy2(source_path, destination_path)
                print(f"Copied: {source_path} -> {destination_path}")

# Contoh penggunaan:
source_dir = os.path.abspath("dataset/image")
output_dir = os.path.abspath("all_data_gambar")
copy_and_rename_images(source_dir, output_dir)
