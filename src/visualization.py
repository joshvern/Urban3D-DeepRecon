#!/usr/bin/env python

import argparse
import numpy as np
import open3d as o3d

def invert_colors(colors):
    """
    Invert colors assuming colors are in the range [0,1].
    """
    return 1.0 - colors

def load_mesh(mesh_path):
    """
    Load the mesh from file.
    """
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if mesh.is_empty():
        raise ValueError(f"Mesh at {mesh_path} is empty or could not be loaded.")
    return mesh

def prepare_mesh(mesh, apply_uniform_color=True, invert=False):
    """
    Ensure the mesh has vertex normals and colors.
    Optionally, apply a uniform color if missing, and invert colors if desired.
    """
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    if not mesh.has_vertex_colors():
        if apply_uniform_color:
            # Apply a uniform light gray color if no colors exist.
            mesh.paint_uniform_color([0.6, 0.6, 0.6])
    else:
        # If colors exist and inversion is requested, invert them.
        if invert:
            colors = np.asarray(mesh.vertex_colors)
            mesh.vertex_colors = o3d.utility.Vector3dVector(invert_colors(colors))
    return mesh

def visualize_mesh(mesh, background_color=[1.0, 1.0, 1.0]):
    """
    Visualize the mesh with a specified background color.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Mesh Visualization', width=1280, height=720, visible=True)
    
    render_option = vis.get_render_option()
    render_option.background_color = np.array(background_color)
    render_option.show_coordinate_frame = True
    
    vis.add_geometry(mesh)
    vis.run()
    vis.destroy_window()

def main():
    parser = argparse.ArgumentParser(
        description="Visualize a 3D mesh with improved options (vertex normals, colors, and background)."
    )
    parser.add_argument("--input", required=True, help="Path to the input mesh file (e.g., PLY, OBJ).")
    parser.add_argument("--invert", action="store_true", help="Invert the vertex colors if they exist.")
    parser.add_argument("--no_uniform", action="store_true", help="Do not apply a uniform color if no colors are present.")
    parser.add_argument("--bg_color", type=float, nargs=3, default=[0.8, 0.8, 0.8],
                        help="Background color as three floats in [0,1] (default is white).")
    args = parser.parse_args()
    
    try:
        mesh = load_mesh(args.input)
    except ValueError as e:
        print(e)
        return

    mesh = prepare_mesh(mesh, apply_uniform_color=not args.no_uniform, invert=args.invert)
    visualize_mesh(mesh, background_color=args.bg_color)

if __name__ == "__main__":
    main()
