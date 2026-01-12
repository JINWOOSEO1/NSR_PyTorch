import viser
import trimesh
import numpy as np
import json
import argparse
import time
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh_path', type=str, default='./source_file/source.ply', help='Path to the source ply file')
    parser.add_argument('--output_path', type=str, default='./source_file/selected_keypoints.json', help='Path to save the selected keypoints')
    args = parser.parse_args()

    if not os.path.exists(args.mesh_path):
        print(f"[Error] Mesh file not found at: {args.mesh_path}")
        print("Please check your working directory or generate the mesh first.")
        return

    print(f"[*] Loading mesh from {args.mesh_path}...")
    mesh = trimesh.load(args.mesh_path)
    
    # Initialize Viser server
    server = viser.ViserServer()
    print(f"[*] Viser server started. Please open the URL displayed above by Viser.")
    
    # Add the mesh to the scene
    server.scene.add_mesh_simple(
        name="/mesh",
        vertices=mesh.vertices,
        faces=mesh.faces,
        color=(200, 200, 200)
    )

    selected_points = []
    point_handles = []

    def save_points():
        with open(args.output_path, 'w') as f:
            json.dump(selected_points, f, indent=4)
        print(f"[*] Saved {len(selected_points)} points to {args.output_path}")

    def add_point_visual(point):
        idx = len(point_handles)
        handle = server.scene.add_point_cloud(
            name=f"/points/p{idx}",
            points=point[None, :],
            colors=np.array([[255, 0, 0]], dtype=np.uint8),
            point_size=0.03,
            point_shape='circle'
        )
        point_handles.append(handle)

    # Load existing points if they exist
    if os.path.exists(args.output_path):
        try:
            with open(args.output_path, 'r') as f:
                loaded_points = json.load(f)
            print(f"[*] Loaded {len(loaded_points)} existing points.")
            for p in loaded_points:
                selected_points.append(p)
                add_point_visual(np.array(p))
        except Exception as e:
            print(f"[!] Failed to load existing points: {e}")

    print("[*] Ready! Click on the mesh to select points.")

    def on_mesh_click(event: viser.ScenePointerEvent):
        # Cast a ray from the camera to find the intersection with the mesh
        origin = event.ray_origin
        direction = event.ray_direction
        
        # Ray-mesh intersection
        locations, index_ray, index_tri = mesh.ray.intersects_location(
            ray_origins=[origin],
            ray_directions=[direction]
        )
        
        if len(locations) > 0:
            # Find closest intersection to the camera
            distances = np.linalg.norm(locations - origin, axis=1)
            closest_idx = np.argmin(distances)
            point = locations[closest_idx]
            
            # Add to list
            selected_points.append(point.tolist())
            add_point_visual(point)
            
            print(f"[+] Added point: {point}")
            save_points()

    # Add a "Clear" button
    gui_clear = server.gui.add_button("Clear All Points")
    
    @gui_clear.on_click
    def _(event):
        for handle in point_handles:
            handle.remove()
        point_handles.clear()
        selected_points.clear()
        save_points()
        print("[-] All points cleared.")

    # Add Selection Mode Toggle
    gui_selection_mode = server.gui.add_checkbox("Selection Mode", initial_value=True)

    def update_selection_mode(_):
        if gui_selection_mode.value:
            # Enable click listener
            server.scene.on_pointer_event('click')(on_mesh_click)
        else:
            # Disable click listener to allow full navigation
            server.scene.remove_pointer_callback()
            
    gui_selection_mode.on_update(update_selection_mode)
    
    # Initialize mode
    update_selection_mode(None)
    print("[-] Added Selection Mode toggle.")

    # Keep the script running
    while True:
        time.sleep(1.0)

if __name__ == "__main__":
    main()
