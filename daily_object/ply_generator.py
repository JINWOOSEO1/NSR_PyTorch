import objaverse
import trimesh
import os
import multiprocessing
import argparse
import numpy as np

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
    target_uids = []
    for uid in lvis_annotations[target_category]:
        target_uids.append(uid)
        if len(target_uids) >= arg.num_objects:
            break
    
    objects = objaverse.load_objects(uids=target_uids)

    cnt = 1
    output_dir = "./target_file/" if arg.is_target else "./source_file/"
    for uid, glb_path in objects.items():
        save_path = os.path.join(output_dir, f"{arg.object}_{cnt}.ply")
        process_glb_to_ply(uid, glb_path, save_path)
        cnt += 1

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    arg = argparse.ArgumentParser()
    arg.add_argument('--is_target', action='store_true', help='If set, output directory is target_file/')
    arg.add_argument('--object', type=str, default='mug', help='Target object category')
    arg.add_argument('--num_objects', type=int, default=5, help='Number of objects to process')
    args = arg.parse_args()

    main(args)