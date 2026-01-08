import os

from datetime import datetime
import numpy as np
import torch
from PIL import Image

from trellis.pipelines import TrellisImageTo3DPipeline
import trellis.models as models
from match_utils.tools import mesh_to_voxels
from match_utils.tools import feature_to_3d, label_voxels_with_colormap, color_ply_with_colormap

os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.

def get_input(mesh_file):
    positions, ss = mesh_to_voxels(mesh_file, coarse_resolution * 4)
    ss = ss[None].float()
    ss = ss.cuda().float()
    latent = encoder(ss, sample_posterior=False)
    assert torch.isfinite(latent).all(), "Non-finite latent"
    return positions, latent

def get_render_imgs(render_file, render_num = 3):
    image_list = []
    for i in range(render_num):
        image_src = os.path.join(render_file, str(i).zfill(3) + '.png')
        image = Image.open(image_src)
        image_list.append(image)
    return image_list

def get_features(images, latent, noise_d = 1, cfg_s = 2.5):
    """Extract features from all network layers.
    Returns a list of features from each transformer block.
    """
    features = pipeline.run_latent_single_step(
        latent=latent,
        images=images,
        extract_time=extract_t,
        noise_d=noise_d,
        sparse_structure_sampler_params={
            "steps": coarse_total_steps,
            "cfg_strength": cfg_s,
        }
    )
    return features

source_dir = 'source_file/'
target_dir = 'target_file/'

pipeline = TrellisImageTo3DPipeline.from_pretrained("pretrained_models/TRELLIS-image-large")
pipeline.cuda()
encoder = models.from_pretrained("pretrained_models/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16").eval().cuda()

# the chosen point on the source ply file
key_point = [0.4, 0.0, 0.0]

output_root_path = os.path.join('output/semantic_match/')
current_time = datetime.now()
formatted_time = current_time.strftime("%m-%d-%H-%M")
output_root_path = os.path.join(output_root_path, formatted_time)

coarse_resolution = 16
coarse_total_steps = 12

# HNSR paper specification: use timestep t=12 for TRELLIS
extract_t = 12
noise_d = 1  # (0-12)

# Hierarchical layer extraction per HNSR paper
# Layer 4: Global/coarse semantic features
# Layers 6, 8, 10: Progressive local refinement
layers_for_matching = [4, 6, 8, 10]



source_image_dir = os.path.join(source_dir, 'renders')
_, source_latent = get_input(source_dir)

source_position, source_coords, source_colormap_dict, source_colormap = label_voxels_with_colormap(source_dir, resolution = coarse_resolution * 4)

distances = np.linalg.norm(np.array(source_position) - np.array(key_point), axis = 1)
chosen_index = np.argmin(distances)
source_occupy_voxels = source_coords // 4
chosen_voxel = source_coords[chosen_index] // 4

chosen_voxel_index = np.where((source_occupy_voxels == chosen_voxel).all(axis = 1))[0]

colormap = np.zeros_like(source_colormap)
colormap[chosen_voxel_index] = [1,1,1] # white color

color_ply_with_colormap(source_position, colormap, output_root_path, name='source.ply')

render_num = 5

source_images = get_render_imgs(source_image_dir, render_num)
source_features = get_features(source_images, source_latent, noise_d=noise_d)

target_image_path =  os.path.join(target_dir, 'renders')
target_positions, target_latent = get_input(target_dir)
target_coords = ((torch.tensor(target_positions) + 0.5) * coarse_resolution * 4).int().contiguous()
target_occupy_voxels = target_coords // 4

target_images = get_render_imgs(target_image_path, render_num)
target_features = get_features(target_images, target_latent, noise_d=noise_d)

# ========== Hierarchical Multi-Layer Semantic Matching ==========
# Implementation following HNSR paper specification
# Uses features from different network layers (4, 6, 8, 10)
# instead of downsampling a single layer

import match_utils.voxel_tool
from match_utils.voxel_tool import voxel_feature_cos_min, voxel_feature_cos_min_coarse_based

# Convert features to proper format for matching
# Each layer's features need to be in (batch, channels, H, W, D) format
print(f"Total layers available: {len(source_features)}")
print(f"Using layers: {layers_for_matching}")

# Step 1: Coarse matching with Layer 4 (global semantic features)
print("\n[Step 1] Coarse matching with Layer 4...")
layer_4_source = feature_to_3d(source_features[layers_for_matching[0]], reso=coarse_resolution)
layer_4_target = feature_to_3d(target_features[layers_for_matching[0]], reso=coarse_resolution)

# Extract feature at chosen voxel location in Layer 4
source_chosen_feature = layer_4_source[0, :, chosen_voxel[0], chosen_voxel[1], chosen_voxel[2]]

# Find coarse matching region in target using Layer 4 features
coarse_mapping = voxel_feature_cos_min(
    target_occupy_voxels,
    layer_4_target,
    source_chosen_feature,
    index_scale=1  # No downscaling, direct voxel matching
)

# Step 2-4: Progressive refinement with Layers 6, 8, 10
for layer_idx in range(1, len(layers_for_matching)):
    current_layer = layers_for_matching[layer_idx]
    prev_layer = layers_for_matching[layer_idx - 1]
    
    print(f"\n[Step {layer_idx + 1}] Refining with Layer {current_layer}...")
    
    # Extract features from current layer
    current_source = feature_to_3d(source_features[current_layer], reso=coarse_resolution)
    current_target = feature_to_3d(target_features[current_layer], reso=coarse_resolution)
    
    # Get feature at chosen point
    source_feature_at_point = current_source[0, :, chosen_voxel[0], chosen_voxel[1], chosen_voxel[2]]
    
    # Refine mapping using previous layer's result as constraint
    coarse_mapping = voxel_feature_cos_min_coarse_based(
        target_occupy_voxels,
        current_target,
        source_feature_at_point,
        index_scale=1,
        coarse_voxel_based=coarse_mapping[0],
        detail_2_coarse_scale=1,  # Same resolution, layer-based refinement
        k=1
    )

# Final result: highlight matched region in target mesh
print("\n[Final] Creating output visualization...")
target_matched_voxel = coarse_mapping[0]
target_map_index = np.where((target_occupy_voxels == target_matched_voxel).all(axis=1))[0]

print(f"Matched Voxel Coordinate: {target_matched_voxel}")
print(f"Number of matched points in point cloud: {len(target_map_index)}")

# Change visualization colors for better visibility
# Base color: Light Gray (so you can see the shape)
target_colormap = np.full((len(target_occupy_voxels), 3), 0.7) 
# Matched color: Bright Red (so it stands out)
target_colormap[target_map_index] = [1, 0, 0]  

# Save result with descriptive filename
output_name = f"matched_t{extract_t}_layers{'_'.join(map(str, layers_for_matching))}.ply"
color_ply_with_colormap(target_positions, target_colormap, output_root_path, name=output_name)

print(f"\nMatching complete! Results saved to: {output_root_path}")
print(f"  - source.ply: Source mesh with white marker at key_point")
print(f"  - {output_name}: Target mesh with RED marker at matched point")
