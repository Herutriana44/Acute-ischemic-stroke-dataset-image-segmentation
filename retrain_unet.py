#!/usr/bin/env python3
"""Entry point: jalankan dari root proyek, mis. `python train_unet.py`."""

from unet_segmentation.retrain_yolo import main

if __name__ == "__main__":
    raise SystemExit(main())