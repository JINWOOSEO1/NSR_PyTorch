import os
os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.

from PIL import Image
import torch as th
from datetime import datetime
from trellis.pipelines import TrellisImageTo3DPipeline
import trellis.models as models

import json

test_file = './shapenet_part/test_file.json'
test_id =  0
with open(test_file, 'r') as f:
    test_params = json.load(f)[test_id]

source_id = test_params.get('source_id')
labels = test_params.get('labels')
category = test_params.get('category')
gt_labels_con = test_params.get('gt_labels_con')

root_path = os.path.join('dataset/ShapeNetCore', category, category)
gt_path = os.path.join('dataset/PartGT', category)

output_root_path = os.path.join('./output/shapenet_part/', category)

test_file_path = os.path.join('test', category, source_id + '.txt')
test_label_path = os.path.join('test', category, source_id + '.json')

current_time = datetime.now()
formatted_time = current_time.strftime("%m-%d-%H-%M")

coarse_resolution = 16
coarse_total_steps = 12

scale_arr = [4, 2, 1]

# output_root_path = os.path.join(output_root_path,source_id, formatted_time)
output_root_path = os.path.join(output_root_path,source_id)

pipeline = TrellisImageTo3DPipeline.from_pretrained("pretrained_models/TRELLIS-image-large")
pipeline.cuda()

encoder = models.from_pretrained("pretrained_models/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16").eval().cuda()

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

def save_features_to_numpy(features_list, output_dir, name='feature.npy'):
    features_list = [feature.cpu().numpy() for feature in features_list]
    features_list = np.array(features_list)
    np.save(os.path.join(output_dir, name), features_list)

source_file = os.path.join(root_path, source_id)
source_mesh_file = os.path.join(source_file, 'models/')
source_image_dir = os.path.join(source_mesh_file, 'renders')
_, source_latent = get_input(source_mesh_file)

from match_utils.tools import label_voxels_with_seg, color_ply

source_position_coords, source_occupy_voxels, source_voxel_labels = label_voxels_with_seg(source_mesh_file, test_label_path, labels, resolution = coarse_resolution * 4) 
color_ply(source_position_coords, source_voxel_labels, range(len(labels)), os.path.join(output_root_path, 'source'))

source_occupy_voxels = source_occupy_voxels // 4

#get source  feature

# best extract_t and extract_l for ShapeNetCorev2
extract_t = 11
extract_l = 9
noise_d = 1
cfg_s = 2.5

if not extract_t is None:
    extract_t_arr = np.array([extract_t])
else:
    extract_t_arr = np.arange(1, coarse_total_steps, 1)

if not extract_l is None:
    extract_l_arr = np.array([extract_l])
else:
    extract_l_arr = np.arange(0, 24, 1)


render_num = 5

output_res = False
saving_features = False
use_his = False
output_iou = False

iou_arr = [[] for _ in range(len(extract_l_arr))]

from match_utils.tools import mapping_occupy_voxels_dict, get_target_2_source_coarse_based, mapping_occupy_voxels_coarse_based, unique_voxels_with_labels, process_predicted_labels, caculate_voxels_IOU_labels

for extract_t in extract_t_arr:
    print('extract_t', str(extract_t), flush=True)

    extract_t = int(extract_t)

    source_images = get_render_imgs(source_image_dir, render_num)

    if use_his and os.path.exists(os.path.join(output_root_path, 'source', str(extract_t) + '.npy')):
        source_features = np.load(os.path.join(output_root_path, 'source', str(extract_t) + '.npy'), allow_pickle=True)
        source_features = [th.tensor(f) for f in source_features]
    else:
        source_features = get_features(source_images, source_latent, extract_t = int(extract_t), noise_d=noise_d, cfg_s = cfg_s)

    if saving_features and not os.path.exists(os.path.join(output_root_path, 'source', str(extract_t) + '.npy')):
        save_features_to_numpy(source_features, output_dir=os.path.join(output_root_path, 'source'), name=str(extract_t) + '.npy')

    test_target_path = np.loadtxt(test_file_path, dtype=str)
    num_target = len(test_target_path)
    print('num_target '+ str(num_target))
    target_path_arr = [os.path.join(root_path, d) for d in test_target_path]
    target_output_arr = [os.path.join(output_root_path, d) for d in test_target_path]
    target_gt_arr = [os.path.join(gt_path, d+'.json') for d in test_target_path]
    test_num = len(target_path_arr)

    total_iou = [0 for _ in range(len(extract_l_arr))]
    totol_num = 0

    for i in range(test_num):
        print('process ', str(i))
        target_mesh_path = os.path.join(target_path_arr[i], 'models')
        target_output_path= target_output_arr[i]
        target_gt_path = target_gt_arr[i]
        target_obj_path = os.path.join(target_mesh_path, 'model_normalized.obj')
        target_image_path =  os.path.join(target_mesh_path, 'renders')
        target_positions, target_latent = get_input(target_mesh_path)
        target_occupy_voxels = ((th.tensor(target_positions) + 0.5) * coarse_resolution * 4).int().contiguous()
        target_occupy_voxels = target_occupy_voxels // 4
        with open(target_gt_path, 'r') as f:
            target_gt_labels = json.load(f).get('labels')
        color_ply(target_positions, target_gt_labels, range(len(labels)), os.path.join(target_output_path), name='gt.ply')

        target_images = get_render_imgs(target_image_path, render_num)

        if use_his and os.path.exists(os.path.join(target_output_path, str(extract_t) + '.npy')):
            target_features = np.load(os.path.join(target_output_path, str(extract_t) + '.npy'), allow_pickle=True)
            target_features = [th.tensor(f) for f in target_features]
        else:
            target_features = get_features(target_images, target_latent, extract_t = extract_t, noise_d=noise_d, cfg_s = cfg_s)

        if saving_features and not os.path.exists(os.path.join(target_output_path, str(extract_t) + '.npy')):
            os.makedirs(target_output_path, exist_ok=True)
            save_features_to_numpy(target_features, target_output_path , name=str(extract_t) + '.npy')

        for l_idx, extract_l in enumerate(extract_l_arr):
            extract_l = int(extract_l)
            source_features_layer = source_features[extract_l]
            source_features_flat_scale_arr = get_down_scale_features(source_features_layer, scale_arr)

            target_features_layer = target_features[extract_l]
            target_features_flat_scale_arr = get_down_scale_features(target_features_layer, scale_arr)

            target_source_coarse_mapping = mapping_occupy_voxels_dict(source_occupy_voxels, target_occupy_voxels, source_features_flat_scale_arr[0], target_features_flat_scale_arr[0], coarse_scale=scale_arr[0])

            for down_idx, down_scale in enumerate(scale_arr):
                if down_idx == 0:continue
                if down_idx == (len(scale_arr) - 1): break
                target_source_coarse_mapping = get_target_2_source_coarse_based(source_occupy_voxels, target_occupy_voxels, source_features_flat_scale_arr[down_idx], target_features_flat_scale_arr[down_idx], target_source_coarse_mapping, coarse_scale=scale_arr[down_idx-1], detail_scale=down_scale)
            
            target_detail_occupy_voxels_labels = mapping_occupy_voxels_coarse_based(source_occupy_voxels, target_occupy_voxels, source_features_flat_scale_arr[-1], target_features_flat_scale_arr[-1], source_voxel_labels, target_source_coarse_mapping, coarse_scale=scale_arr[-2],detail_scale=scale_arr[-1])
            
            _, target_predict_labels = unique_voxels_with_labels(target_occupy_voxels, target_detail_occupy_voxels_labels)
            target_process_predict_labels = process_predicted_labels(target_predict_labels, gt_labels_con)
            target_iou = caculate_voxels_IOU_labels(target_occupy_voxels, target_gt_path, target_process_predict_labels)
            total_iou[l_idx]+=target_iou
            
            print(f'{str(extract_t)}_{str(extract_l)}_{str(cfg_s)}target_iou', target_iou, flush=True)
            if output_res:
                print(f'output ply to {str(extract_t)}_{str(extract_l)}_{str(cfg_s)}.ply', flush=True)
                color_ply(target_positions, target_detail_occupy_voxels_labels, range(len(labels)), target_output_path, name=str(extract_t) +'_' + str(extract_l) + '_' +{str(cfg_s)} +'.ply')
            
        totol_num+=1

    for l_idx in range(len(extract_l_arr)):
        iou_arr[l_idx].append(total_iou[l_idx]/totol_num)
        print(f'average iou{str(l_idx)}_{str(extract_t)}_{str(cfg_s)}', iou_arr[l_idx], flush=True)
    if output_iou:
        print(iou_arr)
        np.save(os.path.join(output_root_path, 'test_iou.npy'), np.array(iou_arr, dtype=object))


# 输出最终结果
for l_idx, iou_res in enumerate(iou_arr):
    print(f'average iou{str(l_idx)}', iou_res, flush=True)

if output_iou:
    np.save(os.path.join(output_root_path, 'test_iou.npy'), np.array(iou_arr))