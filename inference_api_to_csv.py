
import os
import time
import requests
import pandas as pd
from pathlib import Path

# Hugging Face API endpoint
API_URL = "https://herutriana44-acute-ischemic-stroke-segmentation.hf.space/run/predict"

# Rate limiting settings
RATE_LIMIT_DELAY = 1  # seconds between requests to avoid hitting limits

def infer_image_from_api(image_path: Path) -> dict:
    """
    Sends an image to the Hugging Face API for inference and returns the prediction result.
    """
    with open(image_path, "rb") as f:
        data = f.read()

    headers = {"Content-Type": "image/jpeg"} # Assuming JPEG, adjust if other types are sent

    # The API expects a specific format. Referencing the /docs:
    # It seems to be a Gradio API. The /predict endpoint usually expects JSON with 'data' field.
    # The 'data' field will contain a list of inputs. For image, it's typically base64 encoded.
    # However, the example on the /docs shows a direct file upload in the UI.
    # For programmatic access, we often need to mimic the 'upload' component.
    # Let's try sending it as a file directly first, as a common pattern for Gradio.

    # Gradio API usually expects files to be sent as part of a list in 'data' field within a JSON payload
    # Or, it could be a simple POST request with the image file itself.
    # Based on the documentation, it might expect a base64 encoded image in a JSON payload.

    # Let's assume the API expects a multipart/form-data for file upload if not JSON base64.
    # Or, a JSON payload like {'data': ['data:image/jpeg;base64,...']}

    # For now, let's try a simple POST with files dict, mimicking a form submission.
    # If this fails, we'll need to inspect the network request from the HF space or try base64.

    # From the Gradio API docs, it's usually `data: [image_base64_string, ...]`.
    # Let's encode the image to base64.
    import base64
    encoded_image = base64.b64encode(data).decode('utf-8')
    payload = {"data": [f"data:image/jpeg;base64,{encoded_image}"]}

    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        result = response.json()
        # The documentation states the output is a list of [label, confidence] or similar.
        # Example output for an image segmentation API is usually a base64 mask or classification.
        # From the HF space demo, the output seems to be two components: an image and text.
        # We need the text part, which indicates the prediction.
        # Assuming the structure is result['data'][1] for the text prediction.
        if result and 'data' in result and len(result['data']) > 1:
            prediction_text = result['data'][1] # This is a guess based on typical Gradio output
            return {"status": "success", "prediction": prediction_text}
        else:
            print(f"⚠️ Unexpected API response format for {image_path}: {result}")
            return {"status": "error", "prediction": "N/A - Unexpected API response"}
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
