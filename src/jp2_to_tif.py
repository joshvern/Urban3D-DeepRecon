#!/usr/bin/env python

import argparse
import rasterio

def convert_jp2_to_tif(input_path, output_path):
    """
    Convert a JP2 file to a GeoTIFF, preserving georeferencing metadata.
    Ensure that sidecar files (.aux, .j2w, .tab) are in the same directory as the JP2.
    """
    with rasterio.Env():
        with rasterio.open(input_path) as src:
            # Print georeferencing information for verification
            print("Input CRS:", src.crs)
            print("Input Transform:", src.transform)
            
            # Copy the source profile and update the driver to GTiff
            profile = src.profile.copy()
            profile.update(driver='GTiff')
            
            # Write the output GeoTIFF while preserving georeferencing metadata
            with rasterio.open(output_path, 'w', **profile) as dst:
                for i in range(1, src.count + 1):
                    dst.write(src.read(i), i)
                    
    print(f"Converted {input_path} to {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Convert a JP2 file to a GeoTIFF using Rasterio, preserving georeferencing metadata."
    )
    parser.add_argument("-i", "--input", required=True, help="Path to the input JP2 file.")
    parser.add_argument("-o", "--output", required=True, help="Path for the output GeoTIFF file.")
    args = parser.parse_args()
    
    convert_jp2_to_tif(args.input, args.output)

if __name__ == "__main__":
    main()
