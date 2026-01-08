import os
import torch
import numpy as np
import trimesh
from PIL import Image
from easydict import EasyDict as edict
import torch.nn.functional as F
import open3d as o3d
import argparse

from trellis.representations.mesh.cube2mesh import MeshExtractResult
from trellis.renderers.mesh_renderer import MeshRenderer


def get_look_at_matrix(eye, target, up):
    z_axis = F.normalize(eye - target, dim=0)
    x_axis = F.normalize(torch.cross(up, z_axis, dim=0), dim=0)
    y_axis = torch.cross(z_axis, x_axis, dim=0)
    
    T = torch.eye(4, device=eye.device)
    T[:3, 0] = x_axis
    T[:3, 1] = y_axis
    T[:3, 2] = z_axis
    T[:3, 3] = eye
    
    return T # C2W extrinsics

def generate_renders_robust(args):
    if args.mesh_path.split("/")[0] == ".":
        dir = args.mesh_path.split("/")[1]
    else:
        dir = args.mesh_path.split("/")[0]

    output_dir = f"{dir}/renders"
    os.makedirs(output_dir, exist_ok=True)
    mesh_path = args.mesh_path
    device = 'cuda'
    
    print(f"\n[*] Processing: {mesh_path}")

    mesh = trimesh.load(mesh_path)
    
    verts_tensor = torch.from_numpy(mesh.vertices).float().to(device)
    faces_tensor = torch.from_numpy(mesh.faces).long().to(device)
        
    if hasattr(mesh.visual, 'vertex_colors') and len(mesh.visual.vertex_colors) > 0:
        v_color_np = np.array(mesh.visual.vertex_colors[:, :3])
        v_attrs = torch.from_numpy(v_color_np).float().to(device) / 255.0
    else:
        print(f"[!] Warning: {mesh_path} does not have vertex colors. Using default gray.")
        v_attrs = torch.tensor([0.5, 0.5, 0.5], device=device).repeat(verts_tensor.shape[0], 1)
    
    mesh_data = MeshExtractResult(
        vertices=verts_tensor,
        faces=faces_tensor,
        vertex_attrs=v_attrs
    )

    renderer = MeshRenderer(rendering_options={
        "resolution": args.resolution,
        "near": 0.01, 
        "far": 100.0,  
        "ssaa": 2
    }, device=device)

    intrinsics = torch.tensor([
        [args.resolution * 1.0, 0, args.resolution / 2],
        [0, args.resolution * 1.0, args.resolution / 2],
        [0, 0, 1]
    ], device=device).float()

    target = torch.tensor([0, 0, 0], device=device).float()
    up = torch.tensor([0, 1, 0], device=device).float()
    
    success_count = 0
    
    for i in range(args.num_views):
        angle = (2 * np.pi / args.num_views) * i
        
        radius = 3.0 # distance from camera to target in xy plane
        eye_pos = np.array([np.cos(angle) * radius, 3.0, np.sin(angle) * radius])
        eye = torch.from_numpy(eye_pos).float().to(device)
        
        # Extrinsics
        extrinsics = get_look_at_matrix(eye, target, up)
        extrinsics = torch.inverse(extrinsics).float() # C2W -> W2C
        out = renderer.render(mesh_data, extrinsics, intrinsics, return_types=["color", "mask"])
        
        if out["mask"].sum() > 0:
            color_np = (out.color.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255)
            mask_np = out.mask.cpu().numpy() # [H, W]
            
            # Create RGBA for transparency
            img_rgba = np.concatenate([color_np, (mask_np * 255).astype(np.uint8)[..., None]], axis=-1).astype(np.uint8)
            
            save_path = os.path.join(output_dir, f"{str(i).zfill(3)}.png")
            Image.fromarray(img_rgba).save(save_path)
            success_count += 1
        else:
            print(f"[!] Warning: View {i} is empty. Camera at {eye_pos}")

    if success_count == args.num_views:
        print(f"[*] Success: All {args.num_views} views rendered to {output_dir}")
    else:
        print(f"[!] Partial Failure: Only {success_count}/{args.num_views} views rendered.")

if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument('--mesh_path', type=str, default='source_file/mug_1.ply', help='Path to the input PLY mesh file')
    arg.add_argument('--resolution', default = 512, type=int, help='Resolution of the rendered images')
    arg.add_argument('--num_views', default = 5, type=int, help='Number of views to render')
    args = arg.parse_args()
    
    generate_renders_robust(args)