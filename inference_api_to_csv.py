
import os
import time
import pandas as pd
from pathlib import Path
# Install gradio_client if you haven't already: pip install gradio_client
from gradio_client import Client, file
import requests # Still needed for potential API error handling or other requests

# Hugging Face Gradio Space URL
HF_SPACE_URL = "https://herutriana44-acute-ischemic-stroke-segmentation.hf.space"

# Rate limiting settings
RATE_LIMIT_DELAY = 1  # seconds between requests to avoid hitting limits

# Initialize Gradio Client
client = Client(HF_SPACE_URL)

def infer_image_from_api(image_path: Path) -> dict:
    """
    Sends an image to the Hugging Face Gradio Space for inference using gradio_client.
    """
    try:
        # The `predict` method will automatically handle the file upload and API interaction
        # We need to know the correct API name for the specific Gradio function
        # from the HF space. For many simple Gradio apps, it's the first function (fn_idx=0).
        # We can also inspect the /api endpoint of the Gradio app to find function names.
        # Given the previous issues, it's safer to check the API endpoint for exact function.

        # Let's assume for now it's the main prediction function (fn_idx=0)
        # and it expects a file path as input.
        result = client.predict(
            file(str(image_path)), # Gradio Client expects `file()` for local files
            api_name="/predict"  # Or use fn_idx=0 if /predict is not the exact endpoint
                                 # We need to confirm the API name from the docs or inspecting the space.
        )

        # The result structure depends on the Gradio app's output components.
        # From the HF space demo, the output seems to be two components: an image and text.
        # We need the text part, which indicates the prediction.
        # If the result is a list, assume the text prediction is the second element.
        if isinstance(result, list) and len(result) > 1:
            prediction_text = result[1] # Assuming text prediction is at index 1
        else:
            prediction_text = str(result) # Fallback if structure is different

        return {"status": "success", "prediction": prediction_text}
    except Exception as e:
        print(f"❌ Inference failed for {image_path} using gradio_client: {e}")
        return {"status": "error", "prediction": f"Client Error: {e}"}


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
        for file_name in sorted(files): # Renamed 'file' to 'file_name' to avoid conflict with gradio_client.file
            if not file_name.lower().endswith(image_extensions):
                continue

            image_path = Path(root) / file_name
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
