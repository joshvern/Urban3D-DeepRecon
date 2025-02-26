"""
config.py

This file holds configurable parameters and dataset URLs for the project.
"""

import os

# Define the base data directory relative to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Dataset URLs
DATASETS = {
    "lidar": "",  
    "satellite": "" 
}



# Filenames for the datasets after download
FILENAMES = {
    "lidar": os.path.join(DATA_DIR, "sample_lidar.las"),
    "satellite": os.path.join(DATA_DIR, "sample_satellite.tif")
}

# Output mesh file
OUTPUT_MESH = os.path.join(BASE_DIR, "output_mesh.ply")
