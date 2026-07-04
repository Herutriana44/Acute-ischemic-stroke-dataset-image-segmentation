import os
import time
import requests
import pandas as pd
import base64
from pathlib import Path

# Hugging Face Space URL
BASE_URL = "https://herutriana44-acute-ischemic-stroke-segmentation.hf.space"
PREDICT_URL = f"{BASE_URL}/run/predict"
RESULT_URL = f"{BASE_URL}/result"

# Rate limiting settings
RATE_LIMIT_DELAY = 2  # Increased to be safer
WAIT_TIME = 60        # Seconds to wait for async job

def infer_image_from_api(image_path: Path) -> dict:
    """
    Submits image via requests, waits, then polls for result.
    """
    try:
        with open(image_path, "rb") as f:
            data = f.read()

        encoded_image = base64.b64encode(data).decode('utf-8')
        payload = {"data": [f"data:image/png;base64,{encoded_image}"]}

        # 1. Submit Job
        print(f"Submitting {image_path.name}...")
        response = requests.post(PREDICT_URL, json=payload)
        response.raise_for_status()
        job_info = response.json()

        # 2. Wait
        print(f"Waiting {WAIT_TIME}s for job...")
        time.sleep(WAIT_TIME)

        # 3. Get Result
        job_hash = job_info.get("hash")
        if not job_hash:
            return {"status": "error", "prediction": "No job hash returned"}

        res = requests.post(RESULT_URL, json={"hash": job_hash})
        res.raise_for_status()
        result_data = res.json()

        return {"status": "success", "prediction": str(result_data.get("data", "No result"))}

    except Exception as e:
        return {"status": "error", "prediction": str(e)}

def process_images_to_csv(input_folder: Path, output_csv: Path):
    input_folder = Path(input_folder)
    output_csv = Path(output_csv)
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    results = []

    for root, _, files in os.walk(input_folder):
        for file_name in sorted(files):
            if not file_name.lower().endswith(image_extensions):
                continue
            image_path = Path(root) / file_name
            print(f"Processing: {image_path.name}")
            inference_result = infer_image_from_api(image_path)
            results.append({"input_filename": image_path.name, "prediction": inference_result["prediction"]})
            time.sleep(RATE_LIMIT_DELAY)

    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"Done. Saved to {output_csv}")

if __name__ == "__main__":
    process_images_to_csv(Path("all_data_gambar"), Path("inference_api_results.csv"))
