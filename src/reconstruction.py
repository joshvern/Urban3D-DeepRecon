#!/usr/bin/env python

import os
import argparse
import logging
import numpy as np
import open3d as o3d
import pylas

from .config import FILENAMES, OUTPUT_MESH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_point_cloud(lidar_path):
    """
    Load LiDAR data from a LAS file using pylas and convert it to an Open3D point cloud.
    """
    logging.info("Loading point cloud from %s", lidar_path)
    try:
        las = pylas.read(lidar_path)
        points = np.vstack((las.x, las.y, las.z)).transpose()
        
        # If intensity or other channels exist, you can use them to create colors
        # For now, we'll just do a placeholder for intensity if present
        has_intensity = hasattr(las, 'intensity')
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        if pcd.is_empty():
            logging.error("Loaded point cloud is empty.")
            return None
        
        logging.info("Point cloud loaded successfully with %d points.", len(points))
        
        # Example: Assign a grayish color or use intensity as a colormap
        if has_intensity:
            # Normalize intensity to [0,1]
            intensities = las.intensity / float(las.intensity.max())
            # Create an RGB array from intensity (grayscale)
            colors = np.stack([intensities, intensities, intensities], axis=-1)
        else:
            # Uniform gray color for all points
            colors = np.ones((points.shape[0], 3), dtype=np.float32) * 0.6
        
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd
    except Exception as e:
        logging.error("Error loading point cloud: %s", e)
        return None

def filter_point_cloud(pcd, voxel_size=0.5, nb_neighbors=20, std_ratio=2.0):
    """
    Downsample the point cloud and remove outliers.
    """
    logging.info("Downsampling point cloud with voxel size: %f", voxel_size)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    logging.info("Removing statistical outliers (nb_neighbors=%d, std_ratio=%f)", nb_neighbors, std_ratio)
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd

def reconstruct_poisson(pcd, depth=11):
    """
    Perform Poisson surface reconstruction on the point cloud.
    """
    logging.info("Estimating normals for point cloud.")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
    
    logging.info("Performing Poisson reconstruction with depth %d", depth)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    
    # Remove low-density vertices to improve mesh quality
    densities = np.asarray(densities)
    threshold = np.percentile(densities, 1)
    logging.info("Removing vertices with density below the 10th percentile (threshold=%f)", threshold)
    mesh.remove_vertices_by_mask(densities < threshold)
    
    return mesh

def reconstruct_ball_pivoting(pcd):
    """
    Perform mesh reconstruction using the Ball Pivoting Algorithm (BPA).
    """
    logging.info("Estimating normals for BPA.")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
    
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 3 * avg_dist
    logging.info("Performing Ball Pivoting reconstruction with radius: %f", radius)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector([radius, radius * 2])
    )
    return mesh

def save_mesh(mesh, output_path):
    try:
        o3d.io.write_triangle_mesh(output_path, mesh)
        logging.info("Mesh saved to %s", output_path)
    except Exception as e:
        logging.error("Error saving mesh to %s: %s", output_path, e)

def visualize_mesh(mesh):
    """
    Visualize the mesh in Open3D with a white background and uniform color (if desired).
    """
    # If you want to paint the entire mesh a uniform color (e.g., light gray):
    # mesh.paint_uniform_color([0.6, 0.6, 0.6])
    
    # Compute normals for shading if not already computed
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    
    # Create a custom Visualizer to set background color to white
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Open3D', width=1280, height=720, visible=True)
    
    opt = vis.get_render_option()
    opt.background_color = np.array([1.0, 1.0, 1.0])  # White background
    opt.show_coordinate_frame = False
    
    vis.add_geometry(mesh)
    vis.run()
    vis.destroy_window()

def run_reconstruction(lidar_path, output_path, method="poisson", depth=9, voxel_size=0.5, nb_neighbors=20, std_ratio=2.0):
    pcd = load_point_cloud(lidar_path)
    if pcd is None:
        logging.error("No valid point cloud loaded. Exiting reconstruction.")
        return
    
    # Pre-process the point cloud
    pcd = filter_point_cloud(pcd, voxel_size=voxel_size, nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    
    # Choose reconstruction method
    if method == "poisson":
        mesh = reconstruct_poisson(pcd, depth=depth)
    elif method == "bpa":
        mesh = reconstruct_ball_pivoting(pcd)
    else:
        logging.error("Unknown reconstruction method: %s", method)
        return
    
    # Optionally save the mesh
    save_mesh(mesh, output_path)

def main():
    parser = argparse.ArgumentParser(description="LiDAR 3D Reconstruction Pipeline with colored visualization.")
    parser.add_argument('--input', type=str, default=FILENAMES["lidar"],
                        help="Path to the input LiDAR LAS file.")
    parser.add_argument('--output', type=str, default=OUTPUT_MESH,
                        help="Path to save the output 3D mesh.")
    parser.add_argument('--method', type=str, choices=["poisson", "bpa"], default="poisson",
                        help="Reconstruction method to use: 'poisson' or 'bpa'.")
    parser.add_argument('--depth', type=int, default=9,
                        help="Depth parameter for Poisson reconstruction (controls mesh resolution).")
    parser.add_argument('--voxel_size', type=float, default=0.1,
                        help="Voxel size for downsampling the point cloud.")
    parser.add_argument('--nb_neighbors', type=int, default=20,
                        help="Number of neighbors for statistical outlier removal.")
    parser.add_argument('--std_ratio', type=float, default=2.0,
                        help="Standard deviation ratio for outlier removal.")
    args = parser.parse_args()
    
    run_reconstruction(
        lidar_path=args.input,
        output_path=args.output,
        method=args.method,
        depth=args.depth,
        voxel_size=args.voxel_size,
        nb_neighbors=args.nb_neighbors,
        std_ratio=args.std_ratio
    )

if __name__ == "__main__":
    main()
