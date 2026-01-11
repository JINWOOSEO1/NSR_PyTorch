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

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]

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

def radical_inverse(base, n):
    val = 0
    inv_base = 1.0 / base
    inv_base_n = inv_base
    while n > 0:
        digit = n % base
        val += digit * inv_base_n
        n //= base
        inv_base_n *= inv_base
    return val
def halton_sequence(dim, n):
    return [radical_inverse(PRIMES[dim], n) for dim in range(dim)]

def hammersley_sequence(dim, n, num_samples):
    return [n / num_samples] + halton_sequence(dim - 1, n)

def sphere_hammersley_sequence(n, num_samples, offset=(0, 0)):
    u, v = hammersley_sequence(2, n, num_samples)
    u += offset[0] / num_samples
    v += offset[1]
    u = 2 * u if u < 0.25 else 2 / 3 * u + 1 / 3
    theta = np.arccos(1 - 2 * u) - np.pi / 2
    phi = v * 2 * np.pi
    return [phi, theta]
    
def generate_renders_robust(args):
    if args.is_target:
        output_dir = "target_file/renders"
    else:
        output_dir = "source_file/renders"

    os.makedirs(output_dir, exist_ok=True)
    mesh_path = "target_file/target.ply" if args.is_target else "source_file/source.ply"
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
    radius = 2.0 # distance from camera to target in xy plane
    
    success_count = 0
    
    for i in range(args.num_views):
        phi, theta = sphere_hammersley_sequence(i, args.num_views)
        eye_x = radius * np.cos(theta) * np.sin(phi)
        eye_y = radius * np.sin(theta)
        eye_z = radius * np.cos(theta) * np.cos(phi)
        
        eye_pos = np.array([eye_x, eye_y, eye_z])
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
    arg.add_argument('--is_target', action='store_true', help='If set, output directory is target_file/')
    arg.add_argument('--resolution', default = 512, type=int, help='Resolution of the rendered images')
    arg.add_argument('--num_views', default = 10, type=int, help='Number of views to render')
    args = arg.parse_args()
    
    generate_renders_robust(args)