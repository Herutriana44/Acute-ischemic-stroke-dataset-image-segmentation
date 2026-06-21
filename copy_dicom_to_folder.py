import os
import shutil
import uuid

def copy_and_rename_dicom(source_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, dirs, files in os.walk(source_folder):
        for file in files:
            # Sesuaikan ekstensi file DICOM sesuai kebutuhan
            if file.lower().endswith(('.dcm', '.dicom')):
                source_path = os.path.join(root, file)

                # Generate nama acak
                new_filename = str(uuid.uuid4()) + ".dcm"
                destination_path = os.path.join(output_folder, new_filename)

                # Copy file
                shutil.copy2(source_path, destination_path)
                print(f"Copied: {source_path} -> {destination_path}")

# Contoh penggunaan:
source_dir = "dataset"
output_dir = "all_dicom"
copy_and_rename_dicom(source_dir, output_dir)