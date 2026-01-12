import os
from datetime import datetime
import numpy as np
import torch
from PIL import Image
import json
import matplotlib.cm as cm

from trellis.pipelines import TrellisImageTo3DPipeline
import trellis.models as models
from match_utils.tools import mesh_to_voxels, feature_down_sample
from match_utils.tools import feature_to_3d, label_voxels_with_colormap, color_ply_with_colormap
from match_utils.voxel_tool import voxel_feature_cos_min, voxel_feature_cos_min_coarse_based

os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.

def get_input(mesh_file, resolution):
    positions, ss = mesh_to_voxels(mesh_file, resolution)  # positions: (N, 3)
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

def get_features(images, latent, noise_d = 1, cfg_s = 2.5, extract_t=11):
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

# load key points
key_points_path = os.path.join(source_dir, 'selected_keypoints.json')
if os.path.exists(key_points_path):
    print(f"Loading keypoints from {key_points_path}")
    with open(key_points_path, 'r') as f:
        key_points = json.load(f)
else:
    print("There is no keypoint file. Please run keypoint_selector.py first")
    exit()

M = len(key_points) # number of chosen points   
key_points_color = [cm.hsv(i / M)[:3] for i in range(M)]
chosen_voxels = []


output_root_path = os.path.join('output/semantic_match/')
current_time = datetime.now()
formatted_time = current_time.strftime("%m-%d-%H-%M")
output_root_path = os.path.join(output_root_path, formatted_time)

coarse_resolution = 16
coarse_total_steps = 12 # HNSR paper specification: use timestep t=12 for TRELLIS
extract_t = 11
noise_d = 1  # (0-12)
render_num = 5 # number of multi-view images
layers_for_matching = [4, 6, 8, 10]
scale_factor = [4, 4, 2, 1]

source_image_dir = os.path.join(source_dir, 'renders')
_, source_latent = get_input(source_dir, coarse_resolution * 4)
source_position, source_coords, _, source_colormap = label_voxels_with_colormap(source_dir, resolution = coarse_resolution * 4)
source_coords = source_coords // 4 # (N, 3) with resolution 16
source_coords = source_coords.cpu().numpy()

# find the voxel index of the chosen points
colormap = np.ones_like(source_colormap) * 0.7
for p_idx in range(M):
    distance = np.linalg.norm(source_position - key_points[p_idx], axis=1)
    chosen_index = np.argmin(distance)
    chosen_voxel = source_coords[chosen_index]
    chosen_voxels.append(chosen_voxel)
    chosen_voxel_index = np.where((source_coords == chosen_voxel).all(axis = 1))[0]
    # source_colormap[chosen_voxel_index] = key_points_color[p_idx]
    colormap[chosen_voxel_index] = key_points_color[p_idx]

color_ply_with_colormap(source_position, colormap, output_root_path, name='colored_source.ply')

# extract features of source & target
source_images = get_render_imgs(source_image_dir, render_num)
source_features = get_features(source_images, source_latent, noise_d=noise_d, extract_t=extract_t) # 24* (1, 4096, 1024)

target_image_path =  os.path.join(target_dir, 'renders')
_, target_latent = get_input(target_dir, coarse_resolution * 4)
target_positions, target_coords, _, target_colormap = label_voxels_with_colormap(target_dir, resolution = coarse_resolution * 4)
target_coords = target_coords // 4 # (N, 3) with resolution 16

target_images = get_render_imgs(target_image_path, render_num)
target_features = get_features(target_images, target_latent, noise_d=noise_d, extract_t=extract_t) # 24* (1, 4096, 1024)

# find corresponding points
print(f"Total layers available: {len(source_features)}")
print(f"Using layers: {layers_for_matching}")

matching_voxels = np.zeros((M, 3)) # voxel index(16) which correspond to the chosen points in the source 

# Step 1: Global Initialization(layer 4)
source_feature_4 = feature_down_sample(source_features[layers_for_matching[0]], scale_factor[0], coarse_resolution)
target_feature_4 = feature_down_sample(target_features[layers_for_matching[0]], scale_factor[0], coarse_resolution)
layer_4_source = feature_to_3d(source_feature_4, reso=coarse_resolution // scale_factor[0]) # (1, 1024, 4, 4, 4)
layer_4_target = feature_to_3d(target_feature_4, reso=coarse_resolution // scale_factor[0]) # (1, 1024, 4, 4, 4)

for v_idx in range(M):
    chosen_voxel_global = chosen_voxels[v_idx] // scale_factor[0]
    source_chosen_feature = layer_4_source[0, :, chosen_voxel_global[0], chosen_voxel_global[1], chosen_voxel_global[2]]

    coarse_mapping = voxel_feature_cos_min(
        target_coords // scale_factor[0],
        layer_4_target,
        source_chosen_feature,
        index_scale=1
    )
    matching_voxels[v_idx] = coarse_mapping[0]

# Step 2-4: Local Refinement(layer 6,8,10)
for l_idx in range(1, len(layers_for_matching)):
    current_layer = layers_for_matching[l_idx]
    
    source_feature_down = feature_down_sample(source_features[current_layer], scale_factor[l_idx], coarse_resolution)
    target_feature_down = feature_down_sample(target_features[current_layer], scale_factor[l_idx], coarse_resolution)
    current_source = feature_to_3d(source_feature_down, reso=coarse_resolution // scale_factor[l_idx]) 
    current_target = feature_to_3d(target_feature_down, reso=coarse_resolution // scale_factor[l_idx]) 

    for v_idx in range(M):
        chosen_voxel_local = chosen_voxels[v_idx] // scale_factor[l_idx]
        source_feature_at_point = current_source[0, :, chosen_voxel_local[0], chosen_voxel_local[1], chosen_voxel_local[2]]
    
        coarse_mapping = voxel_feature_cos_min_coarse_based(
            target_coords // scale_factor[l_idx],
            current_target,
            source_feature_at_point,
            index_scale=1,
            coarse_voxel_based=matching_voxels[v_idx],
            detail_2_coarse_scale=scale_factor[l_idx-1]//scale_factor[l_idx], 
            k=1
        )
        matching_voxels[v_idx] = coarse_mapping[0]

colormap = np.ones_like(target_colormap) * 0.7
for v_idx in range(M):
    target_map_index = np.where((target_coords == matching_voxels[v_idx]).all(axis=1))[0]
    # target_colormap[target_map_index] = key_points_color[v_idx] 
    colormap[target_map_index] = key_points_color[v_idx]

output_name = f"matched_t_{extract_t}.ply"
color_ply_with_colormap(target_positions, colormap, output_root_path, name=output_name)

print(f"\nMatching complete! Results saved to: {output_root_path}")