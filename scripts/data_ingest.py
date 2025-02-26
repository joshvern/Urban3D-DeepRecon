#!/usr/bin/env python3
"""
data_ingest.py

This script downloads sample LiDAR and satellite imagery data and saves them into the data/ folder.
It uses configurable URLs defined in src/config.py.
"""

import os
import requests
from tqdm import tqdm
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Add the project root to sys.path to import config.py
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
from config import DATA_DIR, DATASETS, FILENAMES

def download_file(url, dest_path):
    """Download a file from a URL with a progress bar."""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()  # Raise an exception for HTTP errors
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        progress = tqdm(total=total_size, unit='iB', unit_scale=True, desc=os.path.basename(dest_path))
        with open(dest_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress.update(len(data))
                file.write(data)
        progress.close()
        if total_size != 0 and progress.n != total_size:
            logging.error("Downloaded file size does not match expected size for %s", dest_path)
        else:
            logging.info("Downloaded %s successfully.", dest_path)
    except Exception as e:
        logging.error("Failed to download %s: %s", url, e)

def ingest_data():
    # Ensure the data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Download LiDAR data
    lidar_url = DATASETS["lidar"]
    lidar_dest = FILENAMES["lidar"]
    logging.info("Downloading LiDAR data from %s", lidar_url)
    download_file(lidar_url, lidar_dest)
    
    # Download Satellite Imagery
    satellite_url = DATASETS["satellite"]
    satellite_dest = FILENAMES["satellite"]
    logging.info("Downloading Satellite Imagery from %s", satellite_url)
    download_file(satellite_url, satellite_dest)
    
    logging.info("Data ingestion complete. Files saved in: %s", DATA_DIR)

if __name__ == "__main__":
    ingest_data()
