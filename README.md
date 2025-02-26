# Urban3D-DeepRecon

Urban3D-DeepRecon is a deep learning–driven pipeline for 3D urban reconstruction that fuses LiDAR point cloud data with satellite imagery. This project demonstrates advanced techniques in semantic segmentation using a U-Net architecture, 3D reconstruction using Open3D, and interactive mesh visualization. It’s designed to showcase skills in handling large geospatial datasets, applying deep learning for image segmentation, and reconstructing detailed 3D models for urban environments.

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Data Ingestion](#data-ingestion)
- [Pipeline Execution](#pipeline-execution)
- [Segmentation Model](#segmentation-model)
- [Exploratory Analysis](#exploratory-analysis)
- [Contributing](#contributing)
- [License](#license)

## Overview

This repository implements a modular pipeline that consists of:

1. **Data Ingestion & Preprocessing:**  
   Downloads LiDAR and satellite imagery datasets using the provided script.
2. **Semantic Segmentation:**  
   Uses a U-Net model to segment satellite imagery. The module supports both training (with a dummy training loop using a pseudo‑mask) and inference.
3. **3D Reconstruction:**  
   Processes LiDAR data with Open3D to generate a 3D mesh of the urban scene.
4. **Visualization:**  
   Provides interactive visualization of the reconstructed 3D mesh.

## Repository Structure

```plaintext
Urban3D-DeepRecon/
├── LICENSE
├── README.md
├── .gitignore
├── requirements.txt
├── main.py
├── data/                      # Raw downloaded data (LiDAR, satellite images)
├── notebooks/                 # Jupyter Notebooks for exploratory data analysis
│   └── exploratory_analysis.ipynb
├── scripts/                   # Utility scripts
│   └── data_ingest.py         # Downloads sample datasets
└── src/                       # Source code for the project
    ├── __init__.py
    ├── config.py            # Configuration parameters and dataset URLs
    ├── segmentation.py      # Segmentation module using U-Net (supports training & inference)
    ├── unet.py              # U-Net architecture implementation
    ├── reconstruction.py    # LiDAR 3D reconstruction module using Open3D
    └── visualization.py     # 3D mesh visualization utilities
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone Urban3D-DeepRecon
   cd Urban3D-DeepRecon
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate 
   ```

3. **Install the dependencies:**
   ```bash
   pip install --upgrade setuptools
   pip install -r requirements.txt
   ```

\## Data Ingestion

Before running the pipeline, download the required datasets (LiDAR and satellite imagery):

```bash
python scripts/data_ingest.py
```

> **Note:** The URLs in `src/config.py` are placeholders. Update them with actual dataset URLs from sources such as [OpenTopography](https://opentopography.org/) for LiDAR data and the [Copernicus Open Access Hub](https://scihub.copernicus.eu/) for satellite imagery.

## Pipeline Execution

You can run the entire pipeline sequentially using the main runner:

```bash
python main.py
```

This will:
- Run the segmentation module on the satellite image.
- Perform 3D reconstruction from the LiDAR data.
- Launch an interactive visualization window to inspect the reconstructed 3D mesh.

Alternatively, run individual modules:

- **Segmentation (Inference):**
  ```bash
  python -m src.segmentation --mode inference --input data/sample_satellite.tif --output segmentation_output.png
  ```
- **Segmentation (Training):**
  ```bash
  python -m src.segmentation --mode train --input data/sample_satellite.tif --epochs 5 --lr 1e-3
  ```
- **3D Reconstruction:**
  ```bash
  python -m src.reconstruction --input data/sample_lidar.las --output output_mesh.ply --depth 9
  ```
- **Visualization:**
  ```bash
  python -m src.visualization --input output_mesh.ply
  ```

## Segmentation Model

The segmentation module uses a U-Net architecture defined in `src/unet.py`. It supports:
- **Training Mode:** A dummy training loop with a pseudo‑mask generated via thresholding the input satellite image.
- **Inference Mode:** Loads pre-trained weights (if available) to perform segmentation and outputs a binary mask.

Update the model training and dataset details as needed for your specific use case.

## Exploratory Analysis

Check out the Jupyter Notebook in the `notebooks/` folder for exploratory analysis of the satellite imagery and LiDAR data. This notebook is a great starting point to understand and visualize your datasets.

## Contributing

Contributions, feedback, and suggestions are welcome! If you have any ideas or improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
