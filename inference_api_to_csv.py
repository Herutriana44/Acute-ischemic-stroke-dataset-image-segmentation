
import os
import time
import requests
import pandas as pd
from pathlib import Path

# Hugging Face API endpoint
API_URL = "https://herutriana44-acute-ischemic-stroke-segmentation.hf.space/inference"

# Rate limiting settings
RATE_LIMIT_DELAY = 1  # seconds between requests to avoid hitting limits

def infer_image_from_api(image_path: Path) -> dict:
    """
    Sends an image to the Hugging Face API for inference and returns the prediction result.
    """
    with open(image_path, "rb") as f:
        data = f.read()

    # The /inference endpoint for many Gradio APIs expects a direct file upload.
    # We will send the image using a 'files' dictionary.
    files = {'file': (image_path.name, data, 'image/png')} # Adjust content type if needed

    try:
        # We remove headers and json payload, and use 'files' instead.
        response = requests.post(API_URL, files=files)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # The response from a direct file upload to a Gradio /inference endpoint
        # might be the direct prediction, not wrapped in a JSON 'data' field.
        # It could be JSON, a file, or plain text depending on the API's output components.
        # Let's try to parse as JSON first, and if not, treat as text.
        try:
            result = response.json()
            # If JSON, we still guess the structure might contain the relevant info.
            # This part is highly dependent on the specific API's return value.
            # If the primary output is a classification or text, it could be a simple value.
            prediction_text = result
        except requests.exceptions.JSONDecodeError:
            # If it's not JSON, it might be plain text.
            prediction_text = response.text

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
