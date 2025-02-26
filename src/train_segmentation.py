#!/usr/bin/env python

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import rasterio
import cv2
from .unet import UNet 

class PatchDataset(Dataset):
    """
    A PyTorch Dataset that extracts patches from a large image and its corresponding mask.
    If no mask is provided, a pseudo-mask is generated using Otsu's thresholding and morphological opening.
    The image and mask are assumed to be single-channel (grayscale) and of the same size.
    """
    def __init__(self, image_path, mask_path=None, patch_size=512, stride=512):
        self.image = self.load_raster(image_path)
        if mask_path is None:
            print("No mask provided. Generating pseudo-mask using thresholding.")
            self.mask = self.generate_pseudo_mask(self.image)
        else:
            self.mask = self.load_raster(mask_path)
        self.patch_size = patch_size
        self.stride = stride
        self.patches = self.extract_patches(self.image, self.mask, patch_size, stride)
    
    def load_raster(self, path):
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            # Normalize to [0,1]
            arr = arr - arr.min()
            if arr.max() > 0:
                arr = arr / arr.max()
            return arr

    def generate_pseudo_mask(self, image):
        """
        Generate a pseudo-mask from the input image using Otsu's thresholding
        and a morphological opening to remove noise.
        """
        # Convert image to uint8
        img_uint8 = (image * 255).astype(np.uint8)
        # Apply Otsu's thresholding
        _, thresh = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Apply morphological opening with a 3x3 kernel to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        pseudo_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        # Normalize mask to [0,1]
        pseudo_mask = pseudo_mask.astype(np.float32) / 255.0
        return pseudo_mask

    def extract_patches(self, image, mask, patch_size, stride):
        """
        Extract patches using a sliding window.
        Returns a list of (patch_image, patch_mask) tuples.
        """
        patches = []
        H, W = image.shape
        for y in range(0, H - patch_size + 1, stride):
            for x in range(0, W - patch_size + 1, stride):
                patch_img = image[y:y+patch_size, x:x+patch_size]
                patch_mask = mask[y:y+patch_size, x:x+patch_size]
                patches.append((patch_img, patch_mask))
        return patches

    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch_img, patch_mask = self.patches[idx]
        # Convert to tensors with shape [1, H, W]
        patch_img = torch.from_numpy(patch_img).unsqueeze(0)
        patch_mask = torch.from_numpy(patch_mask).unsqueeze(0)
        return patch_img, patch_mask

def train_unet(model, dataloader, device, num_epochs=10, lr=1e-3):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")
    
    # Save trained weights
    torch.save(model.state_dict(), "unet_weights.pth")
    print("Training complete. Model weights saved as unet_weights.pth")

def main():
    parser = argparse.ArgumentParser(description="Train U-Net using patch-based training on a large image.")
    parser.add_argument("--image", required=True, help="Path to the input satellite image (JP2/TIFF).")
    parser.add_argument("--mask", required=False, default=None, help="Path to the corresponding ground truth mask (TIF). If not provided, a pseudo-mask will be generated.")
    parser.add_argument("--patch_size", type=int, default=512, help="Size of patches to extract.")
    parser.add_argument("--stride", type=int, default=512, help="Stride for patch extraction.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = PatchDataset(args.image, args.mask, patch_size=args.patch_size, stride=args.stride)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    
    model = UNet(in_channels=1, out_channels=1)
    model.to(device)
    
    train_unet(model, dataloader, device, num_epochs=args.epochs, lr=args.lr)

if __name__ == "__main__":
    main()
