import os
import shutil
import uuid

def is_dicom_file(filepath):
    """Deteksi DICOM berdasarkan magic bytes (byte 128-131: 'DICM')."""
    try:
        with open(filepath, 'rb') as f:
            f.seek(128)
            return f.read(4) == b'DICM'
    except (OSError, IOError):
        return False

def copy_and_rename_dicom(source_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, dirs, files in os.walk(source_folder):
        for file in files:
            source_path = os.path.join(root, file)

            # Cek ekstensi dulu (cepat), lalu magic bytes
            if file.lower().endswith(('.dcm', '.dicom')) or is_dicom_file(source_path):
                new_filename = str(uuid.uuid4()) + ".dcm"
                destination_path = os.path.join(output_folder, new_filename)

                shutil.copy2(source_path, destination_path)
                print(f"Copied: {source_path} -> {destination_path}")

# Contoh penggunaan:
source_dir = os.path.abspath("dataset")
output_dir = os.path.abspath("all_dicom")
copy_and_rename_dicom(source_dir, output_dir)