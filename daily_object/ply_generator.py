import objaverse
import trimesh
import os
import multiprocessing
import argparse
import numpy as np
import json

from daily_object import ply2img

def process_glb_to_ply(uid, glb_path, save_path, angle_offset):
    try:
        loaded = trimesh.load(glb_path)
        if isinstance(loaded, trimesh.Scene):
            mesh = loaded.to_mesh()
        else:
            mesh = loaded
        
        # Normalize mesh
        mesh.apply_translation(-mesh.centroid)
        max_dist = np.max(np.linalg.norm(mesh.vertices, axis=1))
        if max_dist > 0:
            # Important: Scale to 0.45 to fit within the [-0.5, 0.5] voxelization bounds
            mesh.apply_scale(0.45 / max_dist)
        
        # Rotate 90 degrees around X-axis
        rotation_matrix_x = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
        mesh.apply_transform(rotation_matrix_x)

        rotation_matrix_z = trimesh.transformations.rotation_matrix(angle_offset * np.pi / 180.0, [0, 0, 1])
        mesh.apply_transform(rotation_matrix_z)
        
        # Texture를 Vertex Color로 변환
        mesh.visual = mesh.visual.to_color()

        mesh.export(save_path) 
        print(f"[Success] Converted: {uid} -> {save_path}")
        
    except Exception as e:
        print(f"[Error] Failed to process {uid}: {e}")

def generator(arg):
    lvis_annotations = objaverse.load_lvis_annotations()
    target_category = arg.object
    category_uids = lvis_annotations[target_category]
    
    print(f"There are {len(category_uids)} objects in the category: {target_category}")
    select_index = arg.object_idx if arg.object_idx < len(category_uids) else 0 # object index selection
    target_uids = [category_uids[select_index]]
    
    objects = objaverse.load_objects(uids=target_uids)

    start_idx = arg.object_idx if arg.object_idx < len(category_uids) else 0
    save_path = "./target_file/target.ply" if arg.is_target else "./source_file/source.ply"
    
    # Select GLB file
    success = False
    for i in range(start_idx, len(category_uids)):
        uid = category_uids[i]
        objects = objaverse.load_objects(uids=[uid])
        glb_path = objects[uid]
        
        if glb_path.lower().endswith(".glb"):
            print(f"[Selected] Index {i} is a GLB file: {glb_path}")
            process_glb_to_ply(uid, glb_path, save_path, arg.angle)
            success = True
            break
        else:
            print(f"[Skip] Index {i} is not a GLB file (at {glb_path}), trying next...")
            continue
    if not success:
        print("Error: No GLB file found in the remaining category list.")
        return
    
    # Save category & object index in info.json
    info_path = "./target_file/info.json" if arg.is_target else "./source_file/info.json"
    with open(info_path, 'w') as f:
        json.dump({"category": target_category, "index": select_index, "angle":arg.angle}, f, indent = 4, )
        print(f"Saved info.json at {info_path}")

    # Generate renders
    img_save_path = "./target_file/renders" if arg.is_target else "./source_file/renders"
    arg.mesh_path = img_save_path
    arg.resolution = 512
    ply2img.generate_renders_robust(arg)

    # Remove previous voxelization
    voxel_path = "./target_file/voxelize.ply" if arg.is_target else "./source_file/voxelize.ply"
    if os.path.exists(voxel_path):
        os.remove(voxel_path)

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    arg = argparse.ArgumentParser()
    arg.add_argument('--is_target', action='store_true', help='If set, output directory is target_file/')
    arg.add_argument('--object', type=str, default='mug', help='Target object category')
    arg.add_argument('--num_views', type=int, default=10, help='Number of views to render')
    arg.add_argument('--object_idx', type=int, default=0, help='Index of the object to render')
    arg.add_argument('--angle', default = 0, type=float, help='Angle offset for rendering views')
    args = arg.parse_args()
    generator(args)