import numpy as np
import torch as th
import torch.nn.functional as F

def voxel_feature_cos_min(occupy_voxels, target_feature, choose_feature, index_scale):
    target_feature = target_feature.to('cpu').squeeze(0)
    choose_feature = choose_feature.to('cpu')

    grid_index = occupy_voxels // index_scale# vertex_num, 3

    occupy_grid_index = np.unique(grid_index, axis=0)

    feature_map = th.stack([target_feature[:, occupy_index[0], occupy_index[1], occupy_index[2]] for occupy_index in occupy_grid_index], dim=0)

    cos_sim = F.cosine_similarity(feature_map, choose_feature.unsqueeze(0), dim=1)

    sorted_indices = th.argsort(cos_sim, descending=True)  # 从高到低排序的索引
    sorted_voxel_coords = [occupy_grid_index[idx] for idx in sorted_indices]
    return sorted_voxel_coords


def get_detail_voxel_index_coarse_based(occupy_voxels, coarse_voxel_based, coarse_detail_scale=2):
    detail_pick_index = []
    for idx, detail_voxel in enumerate(occupy_voxels):
        detail_voxel_down = np.array(detail_voxel // coarse_detail_scale)
        if np.any(np.all([coarse_voxel_based] == detail_voxel_down, axis=1)):
            detail_pick_index.append(idx)
    return np.array(detail_pick_index)
    
def voxel_feature_cos_min_coarse_based(occupy_voxels, target_feature, choose_feature, index_scale, coarse_voxel_based, detail_2_coarse_scale=2, k =1) :
    target_feature = target_feature.to('cpu').squeeze(0)
    choose_feature = choose_feature.to('cpu')
    occupy_voxels_down = occupy_voxels // index_scale
    occupy_voxels_down = np.unique(occupy_voxels_down, axis=0)
    detail_pick_index = get_detail_voxel_index_coarse_based(occupy_voxels_down, coarse_voxel_based, detail_2_coarse_scale)
    pick_occupy_voxel = occupy_voxels_down[detail_pick_index]

    feature_map = th.stack([target_feature[:, occupy_index[0], occupy_index[1], occupy_index[2]] for occupy_index in pick_occupy_voxel], dim=0)
    cos_sim = F.cosine_similarity(feature_map, choose_feature.unsqueeze(0), dim=1)
    topk_values, topk_indices = th.topk(cos_sim, k=k)
    voxel_coords = []
    for idx in topk_indices:
        voxel_coords.append(pick_occupy_voxel[idx])
    return voxel_coords