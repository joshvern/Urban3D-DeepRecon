#!/usr/bin/env python3
"""
main.py

This script ties together the segmentation, reconstruction, and visualization steps.
It runs the full pipeline sequentially.
"""

import subprocess
import logging
import sys
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def run_command(command, description):
    """Utility function to run a command and log its execution."""
    logging.info("Starting: %s", description)
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        logging.error("Error during: %s", description)
        sys.exit(result.returncode)
    logging.info("Completed: %s", description)

def main():
    # Run segmentation
    run_command("python -m src.segmentation --input data/sample_satellite.tif --output segmentation_output.png",
                "Satellite Image Segmentation")
    #--method sam --model_type vit_b --checkpoint sam_vit_b_01ec64.pth
    # Run 3D reconstruction
    run_command("python -m src.reconstruction --input data/sample_lidar.las --output output_mesh.ply --depth 11",
                "LiDAR 3D Reconstruction")
    
    # Run visualization (this will open an interactive window)
    run_command("python -m src.visualization_texture --mesh output_mesh.ply --tif data/sample_satellite.tif --output textured_mesh.ply --smoothing 2 --decimate 1.0",
                "3D Mesh Visualization")

if __name__ == "__main__":
    main()
