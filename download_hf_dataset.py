import os
from huggingface_hub import snapshot_download

def download_dataset(repo_id, local_dir):
    """
    Downloads a dataset from Hugging Face Hub.
    """
    try:
        print(f"Downloading dataset '{repo_id}' to '{local_dir}'...")
        path = snapshot_download(repo_id=repo_id, local_dir=local_dir, repo_type="dataset")
        print(f"Dataset downloaded successfully to: {path}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")

if __name__ == "__main__":
    REPO_ID = "herutriana44/ais_mri_segmentation_dataset"
    LOCAL_DIR = "./dataset"
    
    # Ensure huggingface_hub is installed: pip install huggingface_hub
    download_dataset(REPO_ID, LOCAL_DIR)
