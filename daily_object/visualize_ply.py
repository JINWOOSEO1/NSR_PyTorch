import viser
import time
import argparse
import open3d as o3d
import numpy as np
import trimesh

def visualization(mesh_path):
    try:
        mesh = trimesh.load(mesh_path)
    except Exception as e:
        print(f"Error loading mesh: {e}")
        return
    
    server = viser.ViserServer(port=8080)
    
    # Determine if it's a Mesh (has faces) or Point Cloud
    if hasattr(mesh, 'faces') and len(mesh.faces) > 0: # 3D mesh
        server.add_mesh_trimesh(
            "/my_mesh",
            mesh=mesh,
            cast_shadow=False,
            receive_shadow=False,
        )
    elif hasattr(mesh, 'vertices'): # Pointcloud
        points = mesh.vertices
        colors = None
        if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
            colors = mesh.visual.vertex_colors[:, :3] 
        
        server.add_point_cloud(
            "/my_point_cloud",
            points=points,
            colors=colors,
            point_size=0.005, 
            point_shape="circle",
        )
    else:
        print("Error: The loaded object has neither faces nor vertices.")

    while True:
        time.sleep(1)

if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument('--mesh_path', default='source_file/source.ply',type=str, required=True, help='Path to the PLY mesh file')
    arg = arg.parse_args()
    
    visualization(arg.mesh_path)  