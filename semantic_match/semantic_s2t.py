import os
os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.

from PIL import Image
import torch as th
from datetime import datetime
from trellis.pipelines import TrellisImageTo3DPipeline
import trellis.models as models


# best extract_t and extract_l for ShapeNetCorev2 co-segmentation
# adjust noise added strength by parameter noise_d for greatly/slightly topology changes
extract_t = 11
extract_l = 9
noise_d = 1 # (0-12)

pipeline = TrellisImageTo3DPipeline.from_pretrained("pretrained_models/TRELLIS-image-large")
pipeline.cuda()

encoder = models.from_pretrained("pretrained_models/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16").eval().cuda()

#chosen source and target file (rendered with official render code)
source_dir = 'source_file'
target_dir = 'target_file'

#the chosen point on the source ply file
point_choose = [0.2702701985836029, 0.16216185688972473, -0.4999999403953552]

output_root_path = os.path.join('output/semantic_match/')

current_time = datetime.now()
formatted_time = current_time.strftime("%m-%d-%H-%M")

coarse_resolution = 16
coarse_total_steps = 12
scale_arr = [4, 2, 1]

output_root_path = os.path.join(output_root_path, formatted_time)

from match_utils.tools import mesh_to_voxels, feature_down_sample, feature_to_3d
import numpy as np

def get_input(mesh_file):
    positions, ss = mesh_to_voxels(mesh_file, coarse_resolution * 4)
    ss = ss[None].float()
    ss = ss.cuda().float()
    latent = encoder(ss, sample_posterior=False)
    assert th.isfinite(latent).all(), "Non-finite latent"
    return positions, latent

def get_render_imgs(render_file, render_num = 3):
    image_list = []
    for i in range(render_num):
        image_src = os.path.join(render_file, str(i).zfill(3) + '.png')
        image = Image.open(image_src)
        image_list.append(image)
    return image_list

def get_features(images, latent, extract_t, noise_d = 1, cfg_s = 2.5):
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

def get_down_scale_features(features, scale_arr):
    features_flat_scale_arr = []
    for down_scale in scale_arr:
        down_scale_features = feature_down_sample(features, down_sample_scale=down_scale)
        features_flat_scale_arr.append(feature_to_3d(down_scale_features, reso=coarse_resolution // down_scale))
    return features_flat_scale_arr

source_image_dir = os.path.join(source_dir, 'renders')
_, source_latent = get_input(source_dir)

from match_utils.tools import label_voxels_with_colormap, color_ply_with_colormap

source_position_coords, source_occupy_voxels_origin, source_voxel_colormap, source_colormap = label_voxels_with_colormap(source_dir, resolution = coarse_resolution * 4)

distances = np.linalg.norm(np.array(source_position_coords) - np.array(point_choose), axis = 1)
chosen_index = np.argmin(distances)
source_occupy_voxels = source_occupy_voxels_origin // 4
choose_voxel = source_occupy_voxels_origin[chosen_index] // 4

chosen_voxel_index = np.where((source_occupy_voxels == choose_voxel).all(axis = 1))[0]

colormap = np.zeros_like(source_colormap)
colormap[chosen_voxel_index] = [1,1,1]
color_ply_with_colormap(source_position_coords, colormap, os.path.join(output_root_path), name='source.ply')

render_num = 5

print('extract_t', str(extract_t), flush=True)
extract_t = int(extract_t)
source_images = get_render_imgs(source_image_dir, render_num)
source_features = get_features(source_images, source_latent, noise_d=noise_d, extract_t = int(extract_t))

target_image_path =  os.path.join(target_dir, 'renders')
target_positions, target_latent = get_input(target_dir)
target_occupy_voxels_origin = ((th.tensor(target_positions) + 0.5) * coarse_resolution * 4).int().contiguous()
target_occupy_voxels = target_occupy_voxels_origin // 4

target_images = get_render_imgs(target_image_path, render_num)
target_features = get_features(target_images, target_latent, noise_d = noise_d, extract_t = extract_t)

extract_l = int(extract_l)
source_features_layer = source_features[extract_l]
source_features_flat_scale_arr = get_down_scale_features(source_features_layer, scale_arr)

target_features_layer = target_features[extract_l]
target_features_flat_scale_arr = get_down_scale_features(target_features_layer, scale_arr)

import match_utils.voxel_tool
import importlib
importlib.reload(match_utils.voxel_tool)
from match_utils.voxel_tool import voxel_feature_cos_min, voxel_feature_cos_min_coarse_based

global_choose_voxel = choose_voxel // scale_arr[0]
global_source_feature = source_features_flat_scale_arr[0][0, : , global_choose_voxel[0], global_choose_voxel[1], global_choose_voxel[2]]
target_source_coarse_mapping = voxel_feature_cos_min(target_occupy_voxels, target_features_flat_scale_arr[0], global_source_feature, scale_arr[0])
print(target_source_coarse_mapping[0])

for down_idx, down_scale in enumerate(scale_arr):
    if down_idx == 0:continue
    local_choose_voxel = choose_voxel // down_scale
    local_source_feature = source_features_flat_scale_arr[down_idx][0, : , local_choose_voxel[0], local_choose_voxel[1], local_choose_voxel[2]]
    target_source_coarse_mapping = voxel_feature_cos_min_coarse_based(target_occupy_voxels, target_features_flat_scale_arr[down_idx], local_source_feature, down_scale,  target_source_coarse_mapping[0], scale_arr[down_idx - 1] // down_scale)
    print(target_source_coarse_mapping[0])
    
target_map_index = np.where(((target_occupy_voxels // down_scale) == target_source_coarse_mapping[0]).all(axis = 1))[0]
target_colormap = np.zeros((len(target_occupy_voxels), 3))
target_colormap[target_map_index] = [1, 1, 1]
color_ply_with_colormap(target_positions, target_colormap, output_root_path, name=str(extract_t) +'_' + str(extract_l) + '_' + str(down_scale)+'.ply')


