import os
import requests
from tqdm import tqdm

# Define constants for your folder and URLs
LOCAL_SAVE_PATH = "data"  # Path where files will be saved
GITHUB_API_URL = "https://api.github.com/repos/chen700564/RGB/contents/data"
RAW_BASE_URL = "https://raw.githubusercontent.com/chen700564/RGB/master/data/"

# Ensure the directory exists before downloading
os.makedirs(LOCAL_SAVE_PATH, exist_ok=True)

def get_file_list():
    """Fetch the list of files from the GitHub repository."""
    response = requests.get(GITHUB_API_URL)
    if response.status_code == 200:
        # Filter files starting with "en" and return their names
        return [file["name"] for file in response.json() if file["name"].startswith("en")]
    else:
        print("Failed to fetch file list:", response.text)
        return []

def file_exists_in_local_directory(file_name):
    """Check if a file with the given name exists in the local directory."""
    local_file_path = os.path.join(LOCAL_SAVE_PATH, file_name)
    return os.path.exists(local_file_path)

def download_file(file_name):
    """Download a file if it doesn't exist locally."""
    if file_exists_in_local_directory(file_name):
        print(f"File already exists: {file_name}. Skipping download.")
        return

    file_url = RAW_BASE_URL + file_name
    local_file_path = os.path.join(LOCAL_SAVE_PATH, file_name)
    response = requests.get(file_url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get("content-length", 0))
        with open(local_file_path, "wb") as file, tqdm(
            total=total_size, unit="B", unit_scale=True, desc=file_name
        ) as progress:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
                progress.update(len(chunk))
        print(f"Downloaded: {file_name}")
    else:
        print(f"Failed to download {file_name}: {response.status_code}")
