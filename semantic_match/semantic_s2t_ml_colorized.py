import os
from datetime import datetime
import numpy as np
import torch
from PIL import Image
import json
import matplotlib.cm as cm
import time

from trellis.pipelines import TrellisImageTo3DPipeline
import trellis.models as models
from match_utils.tools import mesh_to_voxels, feature_down_sample, color_original_mesh_smooth
from match_utils.tools import feature_to_3d, label_voxels_with_colormap
from match_utils.voxel_tool import voxel_feature_cos_min, voxel_feature_cos_min_coarse_based
from daily_object.visualize_ply import visualization

os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.

def shape_matching(pipeline, encoder ,noise_d=1, extract_t=11):
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

    # pipeline = TrellisImageTo3DPipeline.from_pretrained("pretrained_models/TRELLIS-image-large")
    # pipeline.cuda()
    # encoder = models.from_pretrained("pretrained_models/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16").eval().cuda()

    start_time = time.time()

    coarse_resolution = 16
    coarse_total_steps = 12 # HNSR paper specification: use timestep t=12 for TRELLIS
    extract_t = extract_t # 11
    noise_d = noise_d # (0-12)
    render_num = 5 # number of multi-view images
    layers_for_matching = [9,9,9]
    l = len(layers_for_matching)
    scale_factor = []
    for idx in range(l):
        scale_factor.append(2 ** (l-1-idx))

    try:
        with open(os.path.join(source_dir, 'info.json')) as f:
            source_info = json.load(f)
        with open(os.path.join(target_dir, 'info.json')) as f:
            target_info = json.load(f)
    except FileNotFoundError as e:
        print(f"Cannot find json file: {e}")
        source_info = {"category": 'unknown', "index": 'unknown'}
        target_info = {"category": 'unknown', "index": 'unknown'}
    
    layer_str = "".join(map(str, layers_for_matching))
    if source_info['category'] != 'mug':
        output_dir = os.path.join(f"output/visual_correspondence/layer_{layer_str}_{source_info['category']}"
                                  , f"{target_info['category']}_{target_info['index']}")
    elif noise_d != 1:
        if extract_t == 11:
            output_dir = f"./output/visual_correspondence/layer_{layer_str}/{target_info['category']}_{target_info['index']}_rot{target_info['angle']}_noise{noise_d}"
        else:
            output_dir = f"./output/visual_correspondence/layer_{layer_str}/{target_info['category']}_{target_info['index']}_rot{target_info['angle']}_noise{noise_d}_t{extract_t}"

    elif extract_t != 11:
        if noise_d == 1:
            output_dir = f"./output/visual_correspondence/layer_{layer_str}/{target_info['category']}_{target_info['index']}_rot{target_info['angle']}_t{extract_t}"
        else:
            output_dir = f"./output/visual_correspondence/layer_{layer_str}/{target_info['category']}_{target_info['index']}_rot{target_info['angle']}_noise{noise_d}_t{extract_t}"
    elif target_info['angle'] == 0:
        output_dir = f"./output/visual_correspondence/layer_{layer_str}/{target_info['category']}_{target_info['index']}"
    else:
        output_dir = f"./output/visual_correspondence/layer_{layer_str}/{target_info['category']}_{target_info['index']}_rot{target_info['angle']}"
    os.makedirs(output_dir, exist_ok=True)

    source_image_dir = os.path.join(source_dir, 'renders')
    _, source_latent = get_input(source_dir, coarse_resolution * 4)
    source_position, source_coords, source_colormap_dict_64, source_colormap = label_voxels_with_colormap(source_dir, resolution = coarse_resolution * 4)
    source_coords = source_coords // 4 # (N, 3) with resolution 16
    source_coords = source_coords.cpu().numpy()

    source_colormap_dict = {} # (N_source, 3) with resolution 16 -> color
    for k, v in source_colormap_dict_64.items():
        down_k = tuple((np.array(k) // 4).tolist())
        if down_k not in source_colormap_dict:
            source_colormap_dict[down_k] = v

    source_path = os.path.join(source_dir, 'source.ply')
    colored_source_path = os.path.join(output_dir, 'colored_source.ply')
    color_original_mesh_smooth(source_path, source_colormap_dict, colored_source_path)
    print(f"Successfully colored source mesh saved to {colored_source_path}")

    # Extract features of source & target
    source_images = get_render_imgs(source_image_dir, render_num)
    source_features = get_features(source_images, source_latent, noise_d=noise_d, extract_t=extract_t) # 24* (1, 4096, 1024)

    target_image_path =  os.path.join(target_dir, 'renders')
    _, target_latent = get_input(target_dir, coarse_resolution * 4)
    target_positions, target_coords, _, target_colormap = label_voxels_with_colormap(target_dir, resolution = coarse_resolution * 4)
    target_coords = target_coords // 4 # (N_target, 3) with resolution 16

    target_images = get_render_imgs(target_image_path, render_num)
    target_features = get_features(target_images, target_latent, noise_d=noise_d, extract_t=extract_t) # 24* (1, 4096, 1024)

    # Find corresponding points
    print(f"Total layers available: {len(source_features)}")
    print(f"Using layers: {layers_for_matching}")

    N_target = target_coords.shape[0]
    matching_voxels = [{} for _ in range(len(layers_for_matching))]

    matching_time = time.time()

    # Step 1: Global Initialization
    source_feature_g = feature_down_sample(source_features[layers_for_matching[0]], scale_factor[0], coarse_resolution)
    target_feature_g = feature_down_sample(target_features[layers_for_matching[0]], scale_factor[0], coarse_resolution)
    glb_source = feature_to_3d(source_feature_g, reso=coarse_resolution // scale_factor[0]) # (1, 1024, 4, 4, 4)
    glb_target = feature_to_3d(target_feature_g, reso=coarse_resolution // scale_factor[0]) # (1, 1024, 4, 4, 4)

    for v_idx in range(N_target):
        chosen_voxel_global = target_coords[v_idx] // scale_factor[0]
        target_chosen_feature = glb_target[0, :, chosen_voxel_global[0], chosen_voxel_global[1], chosen_voxel_global[2]]
        chosen_voxel_tup = tuple(chosen_voxel_global.tolist())
        if chosen_voxel_tup in matching_voxels[0]:
            continue

        coarse_mapping = voxel_feature_cos_min(
            source_coords // scale_factor[0],
            glb_source,
            target_chosen_feature,
            index_scale=1
        )
        matching_voxels[0][chosen_voxel_tup] = coarse_mapping[0]

    # Step 2: Local Refinement
    for l_idx in range(1, len(layers_for_matching)):
        current_layer = layers_for_matching[l_idx]
        
        source_feature_down = feature_down_sample(source_features[current_layer], scale_factor[l_idx], coarse_resolution)
        target_feature_down = feature_down_sample(target_features[current_layer], scale_factor[l_idx], coarse_resolution)
        current_source = feature_to_3d(source_feature_down, reso=coarse_resolution // scale_factor[l_idx]) 
        current_target = feature_to_3d(target_feature_down, reso=coarse_resolution // scale_factor[l_idx]) 

        for v_idx in range(N_target):
            chosen_voxel_local = target_coords[v_idx] // scale_factor[l_idx]
            target_feature_at_point = current_target[0, :, chosen_voxel_local[0], chosen_voxel_local[1], chosen_voxel_local[2]]
            chosen_voxel_tup = tuple(chosen_voxel_local.tolist())

            if chosen_voxel_tup in matching_voxels[l_idx]:
                continue

            prev_chosen_voxel = target_coords[v_idx]//scale_factor[l_idx-1]
            prev_chosen_voxel_tup = tuple(prev_chosen_voxel.tolist())

            coarse_mapping = voxel_feature_cos_min_coarse_based(
                source_coords // scale_factor[l_idx],
                current_source,
                target_feature_at_point,
                index_scale=1,
                coarse_voxel_based=matching_voxels[l_idx-1][prev_chosen_voxel_tup],
                detail_2_coarse_scale=scale_factor[l_idx-1]//scale_factor[l_idx], 
                k=1
            )
            matching_voxels[l_idx][chosen_voxel_tup] = coarse_mapping[0]

    # Get colors for target voxels(target_coords - (N_target, 3)) from source_voxels(matching_voxels - (N_target, 3))
    target_colormap_dict = {}

    for v_idx in range(N_target):
        matched_voxel = matching_voxels[-1][tuple(target_coords[v_idx].tolist())]
        matched_voxel = tuple(matched_voxel.tolist())
        if matched_voxel in source_colormap_dict:
            color = source_colormap_dict[matched_voxel]
            target_colormap_dict[tuple(target_coords[v_idx].tolist())] = color
        else:
            print(f"There is no corresponding color for voxel: {matched_voxel}")

    target_path = os.path.join(target_dir, 'target.ply')
    colored_target_path = os.path.join(output_dir, "colored_target.ply")

    color_original_mesh_smooth(target_path, target_colormap_dict, colored_target_path)
    print(f"\nMatching complete! Results saved to: {colored_target_path}")

    info_path = os.path.join(output_dir, 'info.txt')
    with open(info_path, 'w') as f:
        f.write(f"Script: {os.path.basename(__file__)}\n")
        f.write(f"The number of views: {render_num}\n")
        f.write(f"Source object: {source_info['category']}_{source_info['index']}\n")
        f.write(f"Source object angle: {source_info['angle']}\n")
        f.write(f"Target object: {target_info['category']}_{target_info['index']}\n")
        f.write(f"Target object angle: {target_info['angle']}\n")
        f.write(f"Layer: {layers_for_matching}\n")
        f.write(f"Scale factor: {scale_factor}\n")
        f.write(f"Extract time: {extract_t}\n")
        f.write(f"Noise D: {noise_d}\n")
        f.write(f"Total time: {time.time() - start_time}\n")
        f.write(f"Extract time: {matching_time - start_time}\n")
        f.write(f"Matching time: {time.time() - matching_time}\n")