import open3d as o3d
import numpy as np

# Path to your .ply file
file_path = './data/raw/egobody/validation/recording_20210910_S05_S06_01_scene_main_01661.ply'

# Read the .ply file
point_cloud = o3d.io.read_point_cloud(file_path)

# Get the points as a numpy array
points = np.asarray(point_cloud.points)

# Print the shape of the point cloud
print(points.shape)


# import open3d as o3d

# def visualize_mesh(mesh_file):
#     # Load the colorized mesh
#     mesh = o3d.io.read_triangle_mesh(mesh_file)
    
#     # Ensure the mesh is correctly loaded and has vertex colors
#     if not mesh.has_vertex_colors():
#         raise ValueError("The mesh does not have vertex colors.")
    
#     # Visualize the mesh
#     o3d.visualization.draw_geometries([mesh])

# if __name__ == '__main__':
#     # Path to the colorized mesh
#     output_mesh_file = 'myevaloutput/pcl_labelled_zed_2.ply'
    
#     # Visualize the output mesh
#     visualize_mesh(output_mesh_file)
