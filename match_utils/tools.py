import os
import numpy as np
import utils3d
import json
from collections import Counter
import trimesh
from scipy.spatial import cKDTree

def voxelize(file, resolution=64):
    import subprocess
    import sys
    
    script = f"""
import os
import sys
import numpy as np
import traceback

resolution = {resolution}
sys.path.append(os.getcwd())

try:
    import open3d as o3d
    import utils3d

    file_path = os.path.abspath(r'{file}')
    model_mesh_path = os.path.join(file_path, 'source.ply')
    if not os.path.exists(model_mesh_path):
        ply_files = [f for f in os.listdir(file_path) if f.endswith('.ply') and f != 'voxelize.ply']
        if ply_files:
            model_mesh_path = os.path.join(file_path, ply_files[0])
        else:
            sys.exit(1)

    mesh = o3d.io.read_triangle_mesh(model_mesh_path)
    if mesh.is_empty():
        sys.exit(1)

    vertices = np.clip(np.asarray(mesh.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    pcd = mesh.sample_points_uniformly(number_of_points=100000)
    
    points = np.asarray(pcd.points)
    indices = np.floor((points + 0.5) * resolution).astype(int)
    indices = np.clip(indices, 0, resolution - 1)
    grid_indices = np.unique(indices, axis=0)
    
    if len(grid_indices) == 0:
        sys.exit(1)
        
    final_vertices = (grid_indices.astype(float) + 0.5) / resolution - 0.5
    utils3d.io.write_ply(os.path.join(file_path, 'voxelize.ply'), final_vertices)

except Exception as e:
    traceback.print_exc()
    sys.exit(1)
"""
    try:
        process = subprocess.run(
            [sys.executable, "-u", "-c", script], 
            capture_output=True, 
            text=True,
            cwd=os.getcwd()
        )
        
        if process.returncode != 0:
            if process.stderr: print(process.stderr.strip())
            raise RuntimeError(f"Voxelization failed for {file}")
            
    except Exception as e:
        raise

def get_voxels(instance, resolution=64):
    import torch
    positions = utils3d.io.read_ply(instance)[0]
    coords = ((torch.tensor(positions) + 0.5) * resolution).int().contiguous()
    ss = torch.zeros(1, resolution, resolution, resolution, dtype=torch.long)
    ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    return positions, ss

def mesh_to_voxels(mesh_dir, resolution=64):
    input_dir = os.path.join(mesh_dir, "voxelize.ply")
    if not os.path.exists(input_dir):
        voxelize(mesh_dir, resolution)

    positions, ss = get_voxels(input_dir, resolution)
    return positions, ss

def label_voxels_with_seg(mesh_dir, label_path,label_list, resolution = 64):
    import torch
    positions = utils3d.io.read_ply(os.path.join(mesh_dir, 'voxelize.ply'))[0]
    coords = ((torch.tensor(positions) + 0.5) * resolution).int().contiguous()
    with open(label_path, 'r') as f:
        data = json.load(f)
    labeled_coords_list = []
    for label in label_list:
        vertex_coords = np.array(data.get(label, None))
        if vertex_coords is None:
            labeled_coords_list.append(np.array([]))
        labeled_coords_list.append(vertex_coords)
    position_labels = []
    for point_c in positions:
        min_distance = float('inf')
        assigned_label = None
        for i, vertex_coords in enumerate(labeled_coords_list):
            distances = np.linalg.norm(vertex_coords - point_c, axis=1)
            closest_distance = np.min(distances)
            if closest_distance < min_distance:
                min_distance = closest_distance
                assigned_label = i
        position_labels.append(assigned_label)
    return positions, coords, position_labels

def feature_down_sample(feature, down_sample_scale = 2, reso= 16):
    import torch
    import torch.nn.functional as F
    feature = feature.to(torch.float32)
    channel = feature.shape[-1]
    feature = feature.view(feature.shape[0], reso,reso,reso, channel)
    downsampled_f = F.avg_pool3d(feature.permute(0, 4, 1, 2, 3), kernel_size=down_sample_scale, stride=down_sample_scale)
    output_tensor = downsampled_f.permute(0, 2, 3, 4, 1).reshape(feature.shape[0], -1, channel)
    return output_tensor

def feature_to_3d(feature, reso=16) : 
    channel = feature.shape[-1]
    feature = feature.view(feature.shape[0], reso,reso,reso, channel).permute(0,4,1,2,3)
    return feature

def mapping_occupy_voxels_dict(source_occupy_voxels, target_occupy_voxels, source_coarse_features, target_coarse_features, coarse_scale=2, threshold=float('inf')):
    import torch
    import torch.nn.functional as F
    source_coarse_features = source_coarse_features.squeeze(0)
    source_occupy_down_voxels = np.unique(np.array(source_occupy_voxels // coarse_scale), axis=0)
    target_occupy_down_voxels = np.unique(np.array(target_occupy_voxels // coarse_scale), axis=0)
    source_occupy_coarse_feature = torch.stack([source_coarse_features[:,occupy_index[0], occupy_index[1], occupy_index[2]] for occupy_index in source_occupy_down_voxels], dim = 0)
    target_source_voxels_mapping_dict = {}
    for voxel_idx, target_down_voxel in enumerate(target_occupy_down_voxels):
        target_coarse_voxel_features = target_coarse_features[0, :, target_down_voxel[0], target_down_voxel[1], target_down_voxel[2]]
        cos_sim = F.cosine_similarity(source_occupy_coarse_feature, target_coarse_voxel_features.unsqueeze(0), dim=1)
        sorted_indices = torch.argsort(cos_sim, descending=True)
        for idx in sorted_indices:
            candidate_source_voxel = source_occupy_down_voxels[idx]
            distance = torch.norm(torch.tensor(candidate_source_voxel, dtype=torch.float32) - torch.tensor(target_down_voxel, dtype=torch.float32), p=2).item()
            if distance <= threshold: 
                target_source_voxels_mapping_dict[tuple(target_down_voxel)] = candidate_source_voxel
                break
            else:
                max_index = torch.argmax(cos_sim)
                target_source_voxels_mapping_dict[tuple(target_down_voxel)] = source_occupy_down_voxels[max_index]
    return target_source_voxels_mapping_dict

def get_index_of_choosing_expand_voxels(target_detail_voxel, source_detail_occupy_voxels, target_source_mapping_dict,scale = 2, expand = 0):
    from scipy.spatial import cKDTree
    target_coarse_voxels = list(target_source_mapping_dict.keys())
    target_coarse_tree = cKDTree(target_coarse_voxels)
    target_detail_voxel_down = target_detail_voxel // scale
    target_coarse_index = target_coarse_tree.query_ball_point(target_detail_voxel_down, r=expand)
    target_coarse_voxel_set = [target_coarse_voxels[i] for i in target_coarse_index]
    source_coarse_voxel_set = np.array([target_source_mapping_dict.get(tuple(target_coarse_pick)) for target_coarse_pick in target_coarse_voxel_set])
    source_detail_pick_index = []
    for idx, source_detail_voxel in enumerate(source_detail_occupy_voxels):
        source_detail_voxel_down = np.array(source_detail_voxel // scale)
        if np.any(np.all(source_coarse_voxel_set == source_detail_voxel_down, axis=1)):
            source_detail_pick_index.append(idx)
    return np.array(source_detail_pick_index)

def get_target_2_source_coarse_based(source_occupy_voxels, target_occupy_voxels, source_coarse_features, target_coarse_features, source_detail_mapping_dict, coarse_scale = 4, detail_scale = 2,expand=1):
    import torch
    import torch.nn.functional as F
    source_coarse_features = source_coarse_features.squeeze(0)
    source_occupy_down_voxels = np.unique(np.array(source_occupy_voxels // detail_scale), axis=0)
    target_occupy_down_voxels = np.unique(np.array(target_occupy_voxels // detail_scale), axis=0)
    source_occupy_down_feature = torch.stack([source_coarse_features[:,occupy_index[0], occupy_index[1], occupy_index[2]] for occupy_index in source_occupy_down_voxels], dim = 0)
    target_source_voxels_mapping_dict = {}
    for voxel_idx, target_down_voxel in enumerate(target_occupy_down_voxels):
        target_occupy_down_voxel_feature = target_coarse_features[0, :, target_down_voxel[0], target_down_voxel[1], target_down_voxel[2]]
        source_voxel_index_pick = get_index_of_choosing_expand_voxels(target_down_voxel, source_occupy_down_voxels,  source_detail_mapping_dict, coarse_scale//detail_scale, expand)
        source_occupy_down_feature_pick = source_occupy_down_feature[source_voxel_index_pick]
        cos_sim = F.cosine_similarity(source_occupy_down_feature_pick, target_occupy_down_voxel_feature.unsqueeze(0), dim=1)
        max_index = torch.argmax(cos_sim)
        target_source_voxels_mapping_dict[tuple(target_down_voxel)] = source_occupy_down_voxels[source_voxel_index_pick[max_index]]
    return target_source_voxels_mapping_dict

def mapping_occupy_voxels_coarse_based(source_occupy_voxels, target_occupy_voxels, source_coarse_features, target_coarse_features, source_occupy_sdf_labels, source_detail_mapping_dict, coarse_scale = 2, detail_scale = 1,expand=1):
    import torch
    import torch.nn.functional as F
    source_coarse_features = source_coarse_features.squeeze(0)
    detail_source_occupy_down_voxels = np.unique(np.array(source_occupy_voxels // detail_scale), axis=0)
    detail_target_occupy_down_voxels = np.unique(np.array(target_occupy_voxels // detail_scale), axis=0)
    source_occupy_detail_feature = torch.stack([source_coarse_features[:,occupy_index[0], occupy_index[1], occupy_index[2]] for occupy_index in detail_source_occupy_down_voxels], dim = 0)
    downsample_source_occupy_labels = get_downsample_labels(source_occupy_voxels, source_occupy_sdf_labels, detail_scale)
    target_voxels_down_label = []
    for voxel_idx, target_down_voxel in enumerate(detail_target_occupy_down_voxels):
        target_detail_voxel_features = target_coarse_features[0, :, target_down_voxel[0], target_down_voxel[1], target_down_voxel[2]]
        source_detail_voxel_index_pick = get_index_of_choosing_expand_voxels(target_down_voxel, detail_source_occupy_down_voxels,  source_detail_mapping_dict, coarse_scale//detail_scale, expand)
        source_occupy_detail_feature_pick = source_occupy_detail_feature[source_detail_voxel_index_pick]
        cos_sim = F.cosine_similarity(source_occupy_detail_feature_pick, target_detail_voxel_features.unsqueeze(0), dim=1)
        max_index = torch.argmax(cos_sim)
        target_voxels_down_label.append(downsample_source_occupy_labels[source_detail_voxel_index_pick[max_index]])
    target_occupy_downsample_voxel_labels = {}
    for idx, target_down_label in enumerate(target_voxels_down_label):
        target_occupy_downsample_voxel_labels[tuple(detail_target_occupy_down_voxels[idx])] = target_down_label
    target_occupy_voxel_labels = []
    for voxel in target_occupy_voxels:
        downsample_voxel = voxel // detail_scale
        downsample_voxel = tuple(downsample_voxel.tolist())
        target_occupy_voxel_labels.append(target_occupy_downsample_voxel_labels.get(downsample_voxel))
    return target_occupy_voxel_labels

def get_downsample_labels(occupy_voxels, occupy_labels, scale):
    occupy_labels = np.array(occupy_labels)
    occupy_voxels = np.array(occupy_voxels)
    downsample_occupy_voxels = np.unique(occupy_voxels // scale ,axis=0)
    downsample_labels = []
    for voxel in downsample_occupy_voxels:
        mask = np.all((occupy_voxels // scale) == voxel, axis=1)
        label_counts = Counter(occupy_labels[mask])
        most_common_label = label_counts.most_common(1)[0][0]
        downsample_labels.append(most_common_label)
    return np.array(downsample_labels)

def color_ply(position_coords, coords_label, label_list, output_file, name='model.ply'):
    vertex_colors = np.ones((len(position_coords), 3))
    colors = color_map[np.array(label_list)]
    for i , label in enumerate(coords_label):
        vertex_colors[i] = colors[label]
    os.makedirs(output_file, exist_ok=True)
    output_path = os.path.join(output_file , name)
    
    # Use utils3d to avoid Open3D/CUDA conflict
    utils3d.io.write_ply(output_path, position_coords, vertex_colors=vertex_colors)
    return True

color_map = np.array([
    [196/255, 151/255, 178/255],
    [241/255, 215/255, 126/255],
    [147/255, 148/255, 231/255],
    [215/255, 99/255, 100/255],
    [177/255, 206/255, 70/255],
    [95/255, 151/255, 210/255],
    [239/255, 122/255, 109/255],
    [99/255, 227/255, 152/255],
    [157/255, 195/255, 231/255]
])
def create_distinct_colormap(voxels, method='hsv'):
    import matplotlib.colors as mcolors
    
    voxels = np.asarray(voxels)
    min_vals = voxels.min(axis=0)
    max_vals = voxels.max(axis=0)
    
    # 1. Normalize to [0, 1]
    normed = (voxels - min_vals) / (max_vals - min_vals + 1e-8)
    
    if method == 'sinusoidal':
        freq = 1.0 
        final_color = 0.5 + 0.5 * np.sin(np.pi * freq * normed)
    elif method == 'hsv':
        h = (normed[:, 0] * 0.5 + normed[:, 1] * 0.3 + normed[:, 2] * 0.2) % 1.0
        s = 0.7 + 0.3 * normed[:, 1] 
        v = 0.8 + 0.2 * normed[:, 2]
        
        hsv = np.stack([h, s, v], axis=1)
        final_color = mcolors.hsv_to_rgb(hsv)
    else:
        raise ValueError("Unknown method")

    colormap_dict = {tuple(voxels[i]): tuple(final_color[i]) for i in range(len(voxels))}
    return colormap_dict, final_color

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return np.array([int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4)])

def create_spatial_colormap(voxels, 
                            start_x='ff0000', end_x='00ffff',   
                            start_y='00ff00', end_y='ff00ff',   
                            start_z='0000ff', end_z='ffff00'):
    voxels = np.asarray(voxels)
    min_vals = voxels.min(axis=0)
    max_vals = voxels.max(axis=0)
    normed = (voxels - min_vals) / (max_vals - min_vals + 1e-8)
    # color_x = (1 - normed[:, 0, None]) * hex_to_rgb(start_x) + normed[:, 0, None] * hex_to_rgb(end_x)
    # color_y = (1 - normed[:, 1, None]) * hex_to_rgb(start_y) + normed[:, 1, None] * hex_to_rgb(end_y)
    # color_z = (1 - normed[:, 2, None]) * hex_to_rgb(start_z) + normed[:, 2, None] * hex_to_rgb(end_z)
    # final_color = (color_x + color_y + color_z) / 3.0
    final_color = 0.55 + 0.45 * normed
    colormap_dict = {tuple(voxels[i]): tuple(final_color[i]) for i in range(len(voxels))}
    return colormap_dict, final_color

def label_voxels_with_colormap(mesh_dir, resolution = 64):
    import torch
    positions = utils3d.io.read_ply(os.path.join(mesh_dir, 'voxelize.ply'))[0]
    coords = ((torch.tensor(positions) + 0.5) * resolution).int().contiguous()
    colormap_dict, final_color = create_distinct_colormap(coords.numpy())
    return positions, coords, colormap_dict, final_color

def upsample_voxel_mapping(
    lowres_mapping_dict, 
    source_voxels_upsampled, 
    target_voxels_upsampled, 
    scale=2
):
    from scipy.spatial import KDTree
    source_voxels_upsampled = np.asarray(source_voxels_upsampled)
    target_voxels_upsampled = np.asarray(target_voxels_upsampled)
    target_voxel_set = set(map(tuple, target_voxels_upsampled))
    lowres_src_keys = np.array(list(lowres_mapping_dict.keys()))
    lowres_src_kdtree = KDTree(lowres_src_keys)
    target_kdtree = KDTree(target_voxels_upsampled)
    upsampled_mapping = {}
    for src_voxel in source_voxels_upsampled:
        coarse_src_voxel = np.floor_divide(src_voxel, scale)
        if tuple(coarse_src_voxel) not in lowres_mapping_dict:
            _, nearest_idx = lowres_src_kdtree.query(coarse_src_voxel)
            coarse_src_voxel = lowres_src_keys[nearest_idx]
        coarse_tgt_voxel = np.array(lowres_mapping_dict[tuple(coarse_src_voxel)])
        residual = src_voxel - np.array(coarse_src_voxel) * scale
        tgt_voxel = coarse_tgt_voxel * scale + residual
        tgt_voxel_tuple = tuple(tgt_voxel.tolist())
        if tgt_voxel_tuple in target_voxel_set:
            upsampled_mapping[tuple(src_voxel)] = tgt_voxel_tuple
        else:
            _, idx = target_kdtree.query(tgt_voxel)
            upsampled_mapping[tuple(src_voxel)] = tuple(target_voxels_upsampled[idx])
    return upsampled_mapping

def color_ply_with_colormap(position_coords, coords_colormap, output_file, name='source.ply'):
    vertex_colors = np.array(coords_colormap)
    os.makedirs(output_file, exist_ok=True)
    output_path = os.path.join(output_file , name)
    
    # Use utils3d to avoid Open3D (which crashes with CUDA initialized)
    utils3d.io.write_ply(output_path, position_coords, vertex_colors=vertex_colors)
    return True

def unique_voxels_with_labels(voxels, labels):
    labels = np.array(labels)
    voxels = np.array(voxels)
    unique_voxels, indices, inverse = np.unique(voxels, axis=0, return_index=True, return_inverse=True)
    unique_labels = np.zeros(len(unique_voxels), dtype=int)
    for i in range(len(unique_voxels)):
        group_labels = labels[inverse == i]
        unique_labels[i] = np.bincount(group_labels).argmax()
    return unique_voxels, unique_labels

def calculate_voxles_iOU(simi_voxels, gt_voxels):
    simi_set = set(map(tuple, simi_voxels))
    gt_set = set(map(tuple, gt_voxels))
    intersection = simi_set.intersection(gt_set)
    union = simi_set.union(gt_set)
    iou = len(intersection) / len(union) if len(union) > 0 else 0.0
    return iou

def process_predicted_labels(predict_labels, gt_labels_cons):
    predict_labels = np.array(predict_labels)
    predict_index = 0
    process_labels = np.zeros_like(predict_labels)
    for label in gt_labels_cons:
        sub_num = label[0]
        gt_label = label[1]
        for i in range(sub_num):
            label_index = predict_index + i
            arr_index = np.argwhere(predict_labels == label_index).reshape(-1)
            process_labels[arr_index] = gt_label
        predict_index += sub_num
    return process_labels

def caculate_voxels_IOU_labels(occupy_voxels, gt_labels_path, predict_labels):
    with open(gt_labels_path, 'r') as f:
        data = json.load(f)
    gt_vertex_labels =np.array(data.get('labels'))
    _, gt_vertex_labels = unique_voxels_with_labels(occupy_voxels, gt_vertex_labels)
    predict_labels = np.array(predict_labels)
    total_iou = 0
    unique_occupy_voxels = np.unique(occupy_voxels, axis=0)
    for label in np.unique(gt_vertex_labels):
        gt_index = np.argwhere(gt_vertex_labels == label).reshape(-1)
        predict_index = np.argwhere(predict_labels == label).reshape(-1)
        iou = calculate_voxles_iOU(unique_occupy_voxels[predict_index], unique_occupy_voxels[gt_index])
        label_ratio = len(gt_index) / len(gt_vertex_labels)
        total_iou += label_ratio * iou
    mean_iou = total_iou
    return mean_iou

# added function for visualization
def color_original_mesh_smooth(mesh_path, colormap_dict, output_path, resolution=16, k=8):
    mesh = trimesh.load(mesh_path, process=False)
    vertices = mesh.vertices
    voxel_coords = np.array(list(colormap_dict.keys())) # (N_voxels, 3), 0~15
    voxel_colors = np.array(list(colormap_dict.values())) # (N_voxels, 3)
    voxel_positions = (voxel_coords + 0.5) / resolution -0.5 # -0.5 ~ 0.5

    tree = cKDTree(voxel_positions)
    dists, idxs = tree.query(vertices, k=k) # (Trilinear와 유사 효과)
    weights = 1.0 / (dists + 1e-8)
    
    weights_sum = weights.sum(axis=1, keepdims=True)
    weights = weights / weights_sum
    vertex_colors = np.sum(weights[:, :, None] * voxel_colors[idxs], axis=1)  # (N_vertex, k, 1) * (N_vertex, k, 3) -> Sum -> (N_vertex, 3)

    if vertex_colors.max() <= 1.0:
        mesh.visual.vertex_colors = (vertex_colors * 255).astype(np.uint8)
    else:
        mesh.visual.vertex_colors = vertex_colors.astype(np.uint8)

    mesh.export(output_path)
    print(f"Saved colored mesh to {output_path}")
    return True