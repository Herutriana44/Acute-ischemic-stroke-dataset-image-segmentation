
import os
import time
import requests
import pandas as pd
import base64
from pathlib import Path

# Hugging Face API endpoint
API_URL = "https://herutriana44-acute-ischemic-stroke-segmentation.hf.space/run/predict"

# Rate limiting settings
RATE_LIMIT_DELAY = 1  # seconds between requests to avoid hitting limits

def infer_image_from_api(image_path: Path) -> dict:
    """
    Sends an image to the Hugging Face API for inference and returns the prediction result.
    """
    try:
        with open(image_path, "rb") as f:
            data = f.read()

        # Base64 encode the image
        encoded_image = base64.b64encode(data).decode('utf-8')

        # Gradio API usually expects JSON with a 'data' field containing inputs.
        # For image, it's typically base64 encoded with a data URI prefix.
        # NOTE: The prefix 'data:image/png;base64,' assumes PNG. Adjust if other formats are needed.
        # For a more robust solution, dynamically determine the image type.
        payload = {"data": [f"data:image/png;base64,{encoded_image}"]}
        headers = {"Content-Type": "application/json"}

        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        result = response.json()

        # The documentation for this specific API, based on the Gradio UI,
        # suggests the output might be in result['data'][1] for the text prediction.
        # If the Gradio app has multiple outputs, they appear as elements in 'data' list.
        if result and 'data' in result and isinstance(result['data'], list) and len(result['data']) > 1:
            prediction_text = result['data'][1]
        else:
            print(f"⚠️ Unexpected API response format for {image_path}: {result}")
            prediction_text = "N/A - Unexpected API response"

        return {"status": "success", "prediction": prediction_text}
    except requests.exceptions.RequestException as e:
        print(f"❌ API request failed for {image_path}: {e}")
        return {"status": "error", "prediction": f"API Error: {e}"}
    except Exception as e:
        print(f"❌ An unexpected error occurred for {image_path}: {e}")
        return {"status": "error", "prediction": f"Unexpected Error: {e}"}


def process_images_to_csv(input_folder: Path, output_csv: Path):
    """
    Processes all image files in `input_folder`, sends them to the API for inference,
    and saves the results to a CSV file.
    """
    input_folder = Path(input_folder)
    output_csv = Path(output_csv)

    if not input_folder.exists():
        raise FileNotFoundError(f"Folder tidak ada: {input_folder.resolve()}")

    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    results = []
    count = 0

    for root, _, files in os.walk(input_folder):
        for file in sorted(files):
            if not file.lower().endswith(image_extensions):
                continue

            image_path = Path(root) / file
            print(f"Processing: {image_path}")

            inference_result = infer_image_from_api(image_path)
            results.append({
                "input_filename": image_path.name,
                "prediction": inference_result["prediction"]
            })
            count += 1
            print(f"✅ Processed {image_path.name} (prediction: {inference_result['prediction']})")

            # Apply rate limiting
            time.sleep(RATE_LIMIT_DELAY)

    if not results:
        print("⚠️ Tidak ada gambar yang berhasil diproses.")
    else:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"📁 Saved inference results to: {output_csv.resolve()} ({count} files) ✅")


if __name__ == "__main__":
    INPUT_FOLDER = Path("all_data_gambar")  # Folder containing images
    OUTPUT_CSV = Path("inference_api_results.csv")

    process_images_to_csv(INPUT_FOLDER, OUTPUT_CSV)
