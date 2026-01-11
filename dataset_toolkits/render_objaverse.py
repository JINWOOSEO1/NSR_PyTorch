import os
import json
import numpy as np
from subprocess import call, DEVNULL
import argparse
import pandas as pd
from utils import sphere_hammersley_sequence
BLENDER_LINK = 'https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz'
BLENDER_INSTALLATION_PATH = '/tmp'
BLENDER_PATH = f'{BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64/blender'

def _install_blender():
    # Already install blender and its dependencies using conda
    conda_prefix = os.environ.get('CONDA_PREFIX')
    env = os.environ.copy()
    if conda_prefix:
        conda_lib_path = os.path.join(conda_prefix, 'lib')
        current_ld_path = env.get('LD_LIBRARY_PATH', '')
        env['LD_LIBRARY_PATH'] = f"{conda_lib_path}:{current_ld_path}"

    if not os.path.exists(BLENDER_PATH):
        os.system('sudo apt-get update')
        os.system('sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6')
        os.system(f'wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}')
        os.system(f'tar -xvf {BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}')
from fnmatch import fnmatch

def _render(file_path, output_dir, num_views):
    yaws = []
    pitchs = []

    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views)
        yaws.append(y)
        pitchs.append(p)
    radius = [2] * num_views
    fov = [40 / 180 * np.pi] * num_views
    views = [{'yaw': y, 'pitch': p, 'radius': r, 'fov': f} for y, p, r, f in zip(yaws, pitchs, radius, fov)]
    print(views)
    args = [
        BLENDER_PATH, '-b', '-P', os.path.join( 'dataset_toolkits/blender_script', 'render.py'), 
        '--',
        '--views', json.dumps(views),
        '--object', os.path.expanduser(file_path),
        '--resolution', '512',
        '--output_folder', output_dir,
        '--engine', 'CYCLES',
        '--save_mesh',
    ]

    call(args, stdout=DEVNULL, stderr=DEVNULL)



down_list = ['cup', 'handbag']
num_views = 5

print('Checking blender...', flush=True)
_install_blender()

for category in down_list:
    print(category)
    obj_root_file = "datasets/Objaverse/" + category
    filter_list = None
    pattern = "*.glb"
    render_obj_list = []
    for path, subdirs, files in os.walk(obj_root_file):
        for name in files:
            dir_name = path.split('/')[-1]
            if fnmatch(name, pattern):
                if os.path.exists(os.path.join(path, 'renders')):
                    continue
                render_obj_list.append((path, name))
    

    for idx, (path, name) in enumerate(render_obj_list):
        os.makedirs(os.path.join(path, 'renders'), exist_ok=True)
        print('Rendering object...',str(idx),str(len(render_obj_list)), flush=True)
        print(os.path.join(path,name))
        _render(os.path.join(path,name), os.path.join(path, 'renders') , num_views)
        print('Render complete.')