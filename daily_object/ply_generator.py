import objaverse
import trimesh
import os
import multiprocessing
import argparse
import numpy as np

from daily_object import ply2img

def process_glb_to_ply(uid, glb_path, save_path):
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
        
        # Texture를 Vertex Color로 변환
        mesh.visual = mesh.visual.to_color()

        mesh.export(save_path) 
        print(f"[Success] Converted: {uid} -> {save_path}")
        
    except Exception as e:
        print(f"[Error] Failed to process {uid}: {e}")

def main(arg):
    lvis_annotations = objaverse.load_lvis_annotations()

    target_category = arg.object
    category_uids = lvis_annotations[target_category]
    
    print(f"There are {len(category_uids)} objects in the category: {target_category}")
    select_index = 15 if len(category_uids) > 15 else 0 # cup: 15, bag: 0
    target_uids = [category_uids[select_index]]
    
    objects = objaverse.load_objects(uids=target_uids)

    save_path = "./target_file/target.ply" if arg.is_target else "./source_file/source.ply"
    for uid, glb_path in objects.items():
        process_glb_to_ply(uid, glb_path, save_path)
        break
    img_save_path = "./target_file/renders" if arg.is_target else "./source_file/renders"
    arg.mesh_path = img_save_path
    arg.resolution = 512
    ply2img.generate_renders_robust(arg)

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    arg = argparse.ArgumentParser()
    arg.add_argument('--is_target', action='store_true', help='If set, output directory is target_file/')
    arg.add_argument('--object', type=str, default='mug', help='Target object category')
    arg.add_argument('--num_views', type=int, default=10, help='Number of views to render')
    args = arg.parse_args()
    main(args)