# Flask Web App - AIS DICOM Segmentation

Web app ini menjalankan inferensi segmentasi stroke ischemic dari data DICOM CT (arsip) dan menampilkan visualisasi 3D.

## Fitur

- Upload archive DICOM: `.zip`, `.rar`, `.tar.gz`, `.tgz`, `.7z`, dan format archive umum lain.
- Auto-detect folder series DICOM (`*.dcm`) dari struktur seperti `0019983/CT`.
- Inference model U-Net dari `best_unet.pt`.
- Visualisasi 3D interaktif:
  - Kontur volume CT (abu-abu)
  - Kontur hasil segmentasi lesi (merah)
- Struktur project web rapi (`templates`, `static`, `uploads`, `runs`, `services`).

## Menjalankan

1. Install dependency:

```bash
pip install -r requirements.txt
```

2. Jalankan Flask app dari folder root project:

```bash
python -m webapp.app
```

3. Buka di browser:

`http://127.0.0.1:5000`

## Catatan

- File upload akan disimpan di `webapp/uploads/`.
- Hasil inferensi per request disimpan di `webapp/runs/<run_id>/`.
- Untuk ekstraksi `.rar`/`.7z`, sistem mungkin membutuhkan tool archive tambahan (mis. `unrar`, `7z`) tergantung environment.
