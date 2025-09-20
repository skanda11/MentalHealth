# data_ingestion/download_gdrive.py
import gdown
import os
from utils.helpers import ensure_dir

def download_file_from_gdrive(gdrive_config):
    """
    Downloads a file from a public Google Drive link using its file ID.
    """
    file_id = gdrive_config.get('gdrive_file_id')
    output_path = gdrive_config.get('output_path')

    if not file_id:
        print("ERROR: Google Drive file ID is not specified in the config file.")
        raise ValueError("gdrive_file_id is missing from the configuration.")

    # Ensure the output directory exists
    ensure_dir(os.path.dirname(output_path))

    print(f"Attempting to download file from Google Drive (ID: {file_id})...")
    
    try:
        gdown.download(id=file_id, output=output_path, quiet=False)
        print(f"Successfully downloaded file to: {output_path}")
    except Exception as e:
        print(f"An error occurred during Google Drive download: {e}")
        print("Please ensure the following:")
        print("1. The 'gdown' library is installed (`pip install gdown`).")
        print("2. The Google Drive file is publicly accessible ('Anyone with the link').")
        print(f"3. The file ID '{file_id}' is correct.")
        raise

if __name__ == '__main__':
    # This block allows you to test the download script directly
    from utils.helpers import load_config
    
    print("Testing Google Drive download script...")
    config = load_config()
    
    if 'gdrive_ingestion' in config:
        download_file_from_gdrive(config['gdrive_ingestion'])
    else:
        print("No 'gdrive_ingestion' section found in config.yaml. Skipping test.")
