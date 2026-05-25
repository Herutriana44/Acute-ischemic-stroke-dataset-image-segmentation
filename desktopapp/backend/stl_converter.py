import os
import dicom2stl
from dicom2stl.main import convert_dicom_to_stl

def convert_dicom_series_to_stl(dicom_dir, output_file, conversion_type="bone"):
    """
    Mengonversi direktori DICOM menjadi file STL.
    
    Args:
        dicom_dir (str): Path ke direktori berisi file DICOM.
        output_file (str): Path file output .stl.
        conversion_type (str): Tipe konversi (bone, skin, dll.)
    """
    if not os.path.exists(dicom_dir):
        raise FileNotFoundError(f"Direktori DICOM tidak ditemukan: {dicom_dir}")

    # Memanggil konverter dicom2stl
    # Menyesuaikan dengan parameter dari README
    convert_dicom_to_stl(
        filenames=[dicom_dir],
        output=output_file,
        type=conversion_type,
        ct=True,
        clean=True,
        verbose=True
    )
    
    return output_file
