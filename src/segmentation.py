#!/usr/bin/env python

import os
import argparse
import numpy as np
import torch
import rasterio
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# Import our custom U-Net (patch inference) and config
from .unet import UNet
from .config import FILENAMES

# Import SAM modules 
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

# Import segmentation models pytorch (SMP)
import segmentation_models_pytorch as smp

#########################
# U-Net Patch Inference #
#########################

def load_satellite_image(path):
    """
    Load the satellite image using rasterio.
    Returns a 2D NumPy array (first band).
    """
    with rasterio.open(path) as src:
        image = src.read(1)
    return image

def patch_segmentation(model, image, patch_size=512, stride=512, device="cpu"):
    """
    Perform segmentation on an image by processing patches.
    Returns a full-size binary segmentation mask.
    """
    H, W = image.shape
    seg_mask = np.zeros((H, W), dtype=np.float32)
    weight_matrix = np.zeros((H, W), dtype=np.float32)
    
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y_end = min(y + patch_size, H)
            x_end = min(x + patch_size, W)
            patch = image[y:y_end, x:x_end]
            
            # If patch is smaller than patch_size, pad it
            ph, pw = patch.shape
            if ph < patch_size or pw < patch_size:
                pad_h = patch_size - ph
                pad_w = patch_size - pw
                patch = np.pad(patch, ((0, pad_h), (0, pad_w)), mode="reflect")
            
            # Normalize and convert patch to tensor
            patch_tensor = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0) / 255.0
            patch_tensor = patch_tensor.to(device)
            
            with torch.no_grad():
                output = model(patch_tensor)
                patch_pred = torch.sigmoid(output)
                patch_pred = (patch_pred > 0.5).float().cpu().numpy().squeeze()
            
            # Remove any padding from the output
            patch_pred = patch_pred[:ph, :pw]
            
            seg_mask[y:y_end, x:x_end] += patch_pred
            weight_matrix[y:y_end, x:x_end] += 1
    
    seg_mask = seg_mask / weight_matrix
    seg_mask = (seg_mask > 0.5).astype(np.float32)
    return seg_mask

def unet_patch_inference_segmentation(satellite_path, output_path, patch_size=512, stride=512):
    """
    Run segmentation using our custom U-Net with patch-based inference.
    """
    device = torch.device("cpu")  # Change to "cuda" if available.
    model = UNet(in_channels=1, out_channels=1)
    model.to(device)
    
    if os.path.exists("unet_weights.pth"):
        model.load_state_dict(torch.load("unet_weights.pth", map_location=device))
        print("Loaded pre-trained U-Net model weights.")
    else:
        print("Warning: Pre-trained weights not found. Running inference with untrained U-Net model.")
    
    model.eval()
    image = load_satellite_image(satellite_path)
    print(f"Satellite image loaded with shape: {image.shape}")
    
    seg_mask = patch_segmentation(model, image, patch_size=patch_size, stride=stride, device=device)
    
    plt.imsave(output_path, seg_mask, cmap="gray")
    print(f"U-Net patch-based segmentation output saved to {output_path}")

#########################
# SMP-based U-Net Inference #
#########################

def smp_inference_segmentation(satellite_path, output_path, patch_size=512, stride=512):
    """
    Run segmentation using a U-Net from segmentation_models_pytorch with pre-trained encoder weights.
    This function performs patch-based inference similar to our custom U-Net.
    """
    device = torch.device("cpu")  # Change to "cuda" if available.
    # Create a U-Net model with a ResNet34 encoder pre-trained on ImageNet
    model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=1, classes=1)
    model.to(device)
    model.eval()
    
    image = load_satellite_image(satellite_path)
    print(f"Satellite image loaded with shape: {image.shape}")
    
    seg_mask = patch_segmentation(model, image, patch_size=patch_size, stride=stride, device=device)
    
    plt.imsave(output_path, seg_mask, cmap="gray")
    print(f"SMP U-Net segmentation output saved to {output_path}")

#########################
# SAM Model Inference   #
#########################

def sam_inference_segmentation(satellite_path, output_path, model_type="vit_h", checkpoint="sam_vit_h_4b8939.pth"):
    """
    Run segmentation using the Segment Anything Model (SAM).
    """
    # Load the image with OpenCV and convert to RGB
    image = cv2.imread(satellite_path)
    if image is None:
        print(f"Error: Could not load image at {satellite_path}")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Initialize SAM using the registry and checkpoint.
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    mask_generator = SamAutomaticMaskGenerator(sam)
    
    # Generate masks; each mask dict contains a "segmentation" key with a binary mask.
    masks = mask_generator.generate(image)
    
    combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for mask in masks:
        combined_mask = np.logical_or(combined_mask, mask["segmentation"]).astype(np.uint8)
    
    plt.imsave(output_path, combined_mask, cmap="gray")
    print(f"SAM segmentation output saved to {output_path}")

#########################
# Command-Line Interface#
#########################

def main():
    parser = argparse.ArgumentParser(
        description="Segmentation script supporting multiple methods: 'unet' (patch inference), 'smp' (pre-trained SMP U-Net), or 'sam'."
    )
    parser.add_argument("--input", required=True, help="Path to the input satellite image.")
    parser.add_argument("--output", required=True, help="Path to save the segmentation output.")
    parser.add_argument("--method", type=str, choices=["unet", "smp", "sam"], default="unet",
                        help="Segmentation method to use: 'unet' (custom U-Net), 'smp' (SMP pre-trained U-Net), or 'sam' (Segment Anything Model).")
    parser.add_argument("--patch_size", type=int, default=512, help="Patch size for U-Net segmentation.")
    parser.add_argument("--stride", type=int, default=512, help="Stride for patch segmentation.")
    parser.add_argument("--model_type", type=str, default="vit_h", help="SAM model type (e.g., vit_h, vit_l, vit_b).")
    parser.add_argument("--checkpoint", type=str, default="sam_vit_h_4b8939.pth",
                        help="Path to the SAM model checkpoint.")
    args = parser.parse_args()
    
    if args.method == "sam":
        sam_inference_segmentation(args.input, args.output, model_type=args.model_type, checkpoint=args.checkpoint)
    elif args.method == "smp":
        smp_inference_segmentation(args.input, args.output, patch_size=args.patch_size, stride=args.stride)
    else:
        unet_patch_inference_segmentation(args.input, args.output, patch_size=args.patch_size, stride=args.stride)

if __name__ == "__main__":
    main()
