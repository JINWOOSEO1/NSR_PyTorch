import viser
import time
import argparse
import open3d as o3d
import numpy as np
import trimesh

def main(mesh_path):
    try:
        mesh = trimesh.load(mesh_path)
    except Exception as e:
        print(f"Error loading mesh: {e}")
        return
    
    server = viser.ViserServer(port=8080)
    server.add_mesh_trimesh(
        "/my_mesh",
        mesh=mesh,
    )

    while True:
        time.sleep(1)  

if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument('--mesh_path', default='source_file/mug_1.ply',type=str, required=True, help='Path to the PLY mesh file')
    arg = arg.parse_args()
    
    main(arg.mesh_path)  