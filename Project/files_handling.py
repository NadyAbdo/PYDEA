import os
import requests
from tqdm import tqdm

# Ensure tqdm is installed
try:
    from tqdm import tqdm
except ImportError:
    os.system('pip install tqdm')

current_directory = os.path.dirname(os.path.abspath(__file__))
files_path = os.path.join(current_directory, "Files")

def download_file(url, save_path):
    # Ensure the folder exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Download the file with progress bar
    with requests.get(url, stream=True) as response:
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB
        print(f"Downloading File...")
        progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)

        with open(save_path, 'wb') as file:
            for data in response.iter_content(chunk_size=block_size):
                progress_bar.update(len(data))
                file.write(data)

        progress_bar.close()

    print(f"File downloaded to: {save_path}")

def main(developer_mode):
    print('Please be patient, to download the dataset this may take a while...')
    if developer_mode == False:
        # URL of the file to download
        file_url = "https://gist.githubusercontent.com/pravalliyaram/5c05f43d2351249927b8a3f3cc3e5ecf/raw/8bd6144a87988213693754baaa13fb204933282d/Mall_Customers.csv"

        # File path to store the downloaded file
        download_path = os.path.join(files_path, "Mall_Customers.csv")

        # Download the file
        download_file(file_url, download_path)

        return os.path.join(files_path, "Mall_Customers.csv")

    if developer_mode == True:
        # For development, return a sample file path
        return os.path.join(files_path, "Mall_Customers.csv")
