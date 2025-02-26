#!/usr/bin/env python

import argparse
import numpy as np
import open3d as o3d
import rasterio
from rasterio.transform import rowcol
from rasterio.enums import Resampling

def load_mesh(mesh_path):
    """
    Load the mesh from a file.
    """
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if mesh.is_empty():
        raise ValueError(f"Mesh at {mesh_path} is empty or could not be loaded.")
    return mesh

def load_orthophoto(tif_path):
    """
    Load the orthophoto from a TIFF file using rasterio.
    Returns the image array and the transform.
    """
    with rasterio.open(tif_path) as src:
        bands = src.count
        # Resample to the native resolution or use bilinear for partial down/up sampling
        image = src.read(
            out_shape=(bands, src.height, src.width),
            resampling=Resampling.bilinear
        )
        # Transpose from (Bands, Height, Width) -> (Height, Width, Bands)
        image = np.transpose(image, (1, 2, 0))
        transform = src.transform
        crs = src.crs
    return image, transform, crs

def sample_color_from_ortho(x, y, image, transform):
    """
    Convert (x, y) in world space to (row, col) in the orthophoto,
    then sample and return a 3-element RGB color in [0,1].
    """
    from rasterio.transform import rowcol

    try:
        row, col = rowcol(transform, x, y)
    except Exception:
        # Fallback to a neutral gray if the transform fails
        return np.array([0.5, 0.5, 0.5], dtype=np.float32)
    
    H, W, C = image.shape
    # If out of bounds, return neutral gray
    if row < 0 or row >= H or col < 0 or col >= W:
        return np.array([0.5, 0.5, 0.5], dtype=np.float32)
    else:
        pixel = image[row, col, :]
        # If RGBA, use only first 3 channels (RGB)
        if C >= 3:
            pixel = pixel[:3]
        elif C == 1:
            # If grayscale, replicate across 3 channels
            pixel = np.array([pixel[0], pixel[0], pixel[0]])
        # Normalize to [0,1]
        return (pixel.astype(np.float32) / 255.0).clip(0, 1)

def apply_texture_to_mesh(mesh, ortho_image, transform):
    """
    Assign vertex colors by sampling the orthophoto at each vertex's (x, y) coordinate.
    """
    vertices = np.asarray(mesh.vertices)
    colors = []
    for v in vertices:
        x, y, z = v
        color = sample_color_from_ortho(x, y, ortho_image, transform)
        colors.append(color)
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(colors))
    return mesh

def refine_mesh(mesh, smoothing_iterations=0, decimate_fraction=1.0, decimate_steps=1):
    """
    Optionally smooth and iteratively decimate the mesh.
    
    :param smoothing_iterations: Number of Laplacian smoothing iterations (0 = no smoothing).
    :param decimate_fraction: Final fraction of triangles to keep (1.0 = no decimation).
    :param decimate_steps: Number of decimation steps to perform.
    """
    if smoothing_iterations > 0:
        print(f"Applying {smoothing_iterations} Laplacian smoothing iteration(s).")
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=smoothing_iterations)
        mesh.compute_vertex_normals()

    if decimate_fraction < 1.0 and decimate_steps > 0:
        current_triangle_count = len(mesh.triangles)
        target_triangle_count = int(current_triangle_count * decimate_fraction)
        print(f"Initial triangle count: {current_triangle_count}")
        print(f"Target triangle count: {target_triangle_count}")

        # Calculate decimation factor per step (multiplicative)
        factor_per_step = (target_triangle_count / current_triangle_count) ** (1.0 / decimate_steps)
        
        for step in range(decimate_steps):
            current_triangle_count = len(mesh.triangles)
            new_target = int(current_triangle_count * factor_per_step)
            if new_target < 4:
                print("Warning: decimation step target is too low; skipping further decimation.")
                break
            print(f"Decimation step {step+1}: reducing from {current_triangle_count} to {new_target} triangles.")
            mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=new_target)
            mesh.compute_vertex_normals()
        print(f"Final triangle count: {len(mesh.triangles)}")
    
    return mesh


def visualize_mesh(mesh, bg_color):
    """
    Visualize the mesh with a specified background color.
    """
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Textured Mesh', width=1280, height=720, visible=True)
    
    render_option = vis.get_render_option()
    render_option.background_color = np.array(bg_color)
    render_option.show_coordinate_frame = True
    
    vis.add_geometry(mesh)
    vis.run()
    vis.destroy_window()

def main():
    parser = argparse.ArgumentParser(
        description="Visualize a 3D mesh textured with an orthophoto (TIFF), with optional smoothing & iterative decimation."
    )
    parser.add_argument("--mesh", required=True, help="Path to the input 3D mesh file (e.g., PLY, OBJ).")
    parser.add_argument("--tif", required=True, help="Path to the orthophoto TIFF file.")
    parser.add_argument("--output", required=False, help="(Optional) Path to save the textured mesh.")
    parser.add_argument("--smoothing", type=int, default=2,
                        help="Number of Laplacian smoothing iterations (0 = no smoothing).")
    parser.add_argument("--decimate", type=float, default=0.8,
                        help="Final fraction of triangles to keep (1.0 = no decimation, 0.8 = keep 80%%, etc.).")
    parser.add_argument("--decimate_steps", type=int, default=3,
                        help="Number of iterative decimation steps.")
    parser.add_argument("--bg_color", type=float, nargs=3, default=[0.8, 0.8, 0.8],
                        help="Background color as three floats in [0,1].")
    args = parser.parse_args()
    
    # Load mesh
    mesh = load_mesh(args.mesh)
    
    # Load orthophoto
    ortho_image, transform, crs = load_orthophoto(args.tif)
    print("Orthophoto loaded. Image shape:", ortho_image.shape)
    print("Transform:", transform)
    print("CRS:", crs)
    
    # Apply texture to the mesh
    textured_mesh = apply_texture_to_mesh(mesh, ortho_image, transform)
    
    # Refine mesh with iterative decimation
    textured_mesh = refine_mesh(
        textured_mesh,
        smoothing_iterations=args.smoothing,
        decimate_fraction=args.decimate,
        decimate_steps=args.decimate_steps
    )
    
    # Optionally save the textured mesh
    if args.output:
        o3d.io.write_triangle_mesh(args.output, textured_mesh)
        print(f"Textured mesh saved to {args.output}")
    
    # Visualize the textured mesh
    visualize_mesh(textured_mesh, args.bg_color)

if __name__ == "__main__":
    main()
