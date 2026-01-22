import os
from datetime import datetime
import numpy as np
import torch
from PIL import Image
import json
import matplotlib.cm as cm
import time
import importlib
import argparse

os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'    

from trellis.pipelines import TrellisImageTo3DPipeline
import trellis.models as models
from match_utils.tools import mesh_to_voxels, feature_down_sample, color_original_mesh_smooth
from match_utils.tools import feature_to_3d, label_voxels_with_colormap
from match_utils.voxel_tool import voxel_feature_cos_min, voxel_feature_cos_min_coarse_based
from daily_object.visualize_ply import visualization
import semantic_match.semantic_s2t_ml as s2t_ml
import semantic_match.semantic_s2t_ml_colorized as s2t_ml_colorized
import daily_object.ply_generator as pl


pipeline = TrellisImageTo3DPipeline.from_pretrained("pretrained_models/TRELLIS-image-large")
pipeline.cuda()
encoder = models.from_pretrained("pretrained_models/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16").eval().cuda()

def main():
    object_and_index = {
        'mug': [15, 8],
        'pitcher_(vessel_for_liquid)': [1, 11],
        'coffeepot': [0],
        'kettle': [29],
        'basket': [5, 20],
        'handbag': [0]
    }
    while True:
        command = input("type rotate to check rotation!! if not various test cases will be executed q to quit: ")
        importlib.reload(s2t_ml)
        importlib.reload(s2t_ml_colorized)
        importlib.reload(pl)
        if command == 'q':
            break
        elif command == 'rotate':
            try:
                # rotation
                for angle in range(0, 360, 45):
                    print(f"Processing rotation angle: {angle} degrees")
                    arg = argparse.ArgumentParser()
                    arg.is_target = True
                    arg.object = 'mug'
                    arg.object_idx = 8
                    arg.num_views = 5
                    arg.angle = angle
                    pl.generator(arg)

                    s2t_ml.point_matching(pipeline, encoder, noise_d=10, extract_t=10)
                    # s2t_ml_colorized.shape_matching(pipeline, encoder)
            except Exception as e:
                print("Error during matching: ", e)
        elif command == 'test':
            try:
                # various test cases
                for object, idx_ls in object_and_index.items():
                    for idx in idx_ls:
                        print(f"Processing object: {object} with index: {idx}")
                        arg = argparse.ArgumentParser()
                        arg.is_target = True
                        arg.object = object
                        arg.object_idx = idx
                        arg.num_views = 5
                        arg.angle = 0.0
                        pl.generator(arg)

                        s2t_ml.point_matching(pipeline, encoder)
                        s2t_ml_colorized.shape_matching(pipeline, encoder)
            except Exception as e:
                print("Error during matching: ", e)
        elif command == 'noise':
            try:
                for angle in [45, 90, 135, 180]:
                    for noise_d in range(2, 12, 1):
                        print(f"Processing noise: {noise_d}")
                        arg = argparse.ArgumentParser()
                        arg.is_target = True
                        arg.object = 'mug'
                        arg.object_idx = 8
                        arg.num_views = 5
                        arg.angle = angle
                        pl.generator(arg)

                        s2t_ml.point_matching(pipeline, encoder, noise_d, extract_t=9) 
                        # s2t_ml_colorized.shape_matching(pipeline, encoder, noise_d)
            except Exception as e:
                print("Error during matching: ", e)
        elif command == 'extract_t':
            try:
                for angle in [45, 90, 135, 180]:
                    for t in range(1, 11, 1):
                        print(f"Processing extract time: {t}")
                        arg = argparse.ArgumentParser()
                        arg.is_target = True
                        arg.object = 'mug'  
                        arg.object_idx = 8
                        arg.num_views = 5
                        arg.angle = angle
                        pl.generator(arg)

                        s2t_ml.point_matching(pipeline, encoder, noise_d=10, extract_t=t) 
                        #s2t_ml_colorized.shape_matching(pipeline, encoder, extract_t=t)
            except Exception as e:
                print("Error during matching: ", e)
        else:
            print("invalide input")

if __name__ == '__main__':
    main()