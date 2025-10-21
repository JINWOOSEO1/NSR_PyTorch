# [SIGGRAPH ASIA 2025] Hierarchical Neural Semantic Representation for 3D Semantic Correspondence

<div align="center">

**Keyu Du<sup>1</sup>, Jingyu Hu<sup>2†</sup>, Haipeng Li<sup>1</sup>, Hao Xu<sup>2</sup>, Haibing Huang<sup>3</sup>, Chi-Wing Fu<sup>2</sup>, Shuaicheng Liu<sup>1</sup>**

<sup>1</sup>University of Electronic Science and Technology of China (UESTC)
<sup>2</sup>The Chinese University of Hong Kong (CUHK)
<sup>3</sup>TeleAI

† Corresponding author  

</div>

This is the official implementation of our SIGGRAPH ASIA 2025 paper, [Hierarchical Neural Semantic Representation for 3D Semantic Correspondence](https://arxiv.org/abs/2509.17431).

## 💡 Introduction

This paper presents a new approach to estimate accurate and robust 3D semantic correspondence with hierarchical neural semantic representation.

Our work has three key contributions:

1. **Hierarchical Neural Semantic Representation (HNSR)**: We design a representation that consists of a global semantic feature to capture high-level structure and multi-resolution local geometric features to preserve fine details, by carefully harnessing 3D priors from pre-trained 3D generative models.

2. **Progressive Global-to-Local Matching Strategy**: We design a strategy that establishes coarse semantic correspondence using the global semantic feature, then iteratively refines it with local geometric features, yielding accurate and semantically-consistent mappings.

3. **Training-Free Framework**: Our framework is training-free and broadly compatible with various pre-trained 3D generative backbones, demonstrating strong generalization across diverse shape categories.

Our method also supports various applications, such as shape co-segmentation, keypoint matching, and texture transfer, and generalizes well to structurally diverse shapes, with promising results even in cross-category scenarios. Both qualitative and quantitative evaluations show that our method outperforms previous state-of-the-art techniques.

![Teaser Image](./assets/teaser.png)


## 📌 To-Do List
- [x] Release NSR-PyTorch code with pretrained model, environment setup guide, dataset preparation, and batch testing scripts
- [x] Batch testing script for Objaverse dense matching  
- [x] Batch testing script for ShapeNet co-segmentation  
- [x] Demo code for other backbones [NSR_LAS_PyTorch](https://github.com/mapledky/NSR_LAS_PyTorch).
- [ ] Single source-to-target demo script  
- [ ] Keypoint matching testing script  

This repository is still under active development.  
If you have any questions or issues, feel free to contact me at [Email](mailto:dukeyu@std.uestc.edu.cn).


<!-- Installation -->
## 🚀 Installation

### Prerequisites

- **System**: The code is currently tested only on **Linux**.
- **Hardware**: An NVIDIA GPU with at least 24GB of memory is necessary. The code has been verified on Tesla P40.  
- **Software**:   
  - The [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) is needed to compile certain submodules. The code has been tested with CUDA versions 11.8 and 12.2.  
  - [Conda](https://docs.anaconda.com/miniconda/install/#quick-command-line-install) is recommended for managing dependencies.  
  - Python version 3.8 or higher is required. 


###Installation Steps

1. **Clone the repository:**
    ```bash
    git clone --recurse-submodules https://github.com/mapledky/NSR_PyTorch.git
    cd NSR_PyTorch
    ```

2. **Install the dependencies:**
    
    **Important notes before running the installation command:**
    - By adding `--new-env`, a new conda environment named `nsr` will be created. If you want to use an existing conda environment, please remove this flag.
    - By default, the `nsr` environment will use PyTorch 2.4.0 with CUDA 11.8. If you want to use a different version of CUDA (e.g., if you have CUDA Toolkit 12.2 installed and do not want to install another 11.8 version for submodule compilation), you can remove the `--new-env` flag and manually install the required dependencies. Refer to [PyTorch](https://pytorch.org/get-started/previous-versions/) for the installation command.
    - If you have multiple CUDA Toolkit versions installed, `PATH` should be set to the correct version before running the command. For example, if you have CUDA Toolkit 11.8 and 12.2 installed, you should run `export PATH=/usr/local/cuda-11.8/bin:$PATH` before running the command.
    - By default, the code uses the `flash-attn` backend for attention. For GPUs that do not support `flash-attn` (e.g., NVIDIA V100), you can remove the `--flash-attn` flag to install `xformers` only and set the `ATTN_BACKEND` environment variable to `xformers` before running the code.
    - The installation may take a while due to the large number of dependencies. Please be patient. If you encounter any issues, you can try to install the dependencies one by one, specifying one flag at a time.
    
    **Create a new conda environment named `nsr` and install the dependencies:**
    ```bash
    . ./setup.sh --new-env --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast
    ```
    
    **Detailed usage of `setup.sh`:**
    ```bash
    . ./setup.sh --help
    ```
    
    **Available options:**
    ```
    Usage: setup.sh [OPTIONS]
    Options:
        -h, --help              Display this help message
        --new-env               Create a new conda environment
        --basic                 Install basic dependencies
        --train                 Install training dependencies
        --xformers              Install xformers
        --flash-attn            Install flash-attn
        --diffoctreerast        Install diffoctreerast
        --spconv                Install spconv
        --mipgaussian           Install mip-splatting
        --kaolin                Install kaolin
        --nvdiffrast            Install nvdiffrast
    ```

<!-- Pretrained Models -->
## 🤖 Pretrained Models

We provide the following pretrained model following the official [microsoft/TRELLIS](https://github.com/microsoft/TRELLIS):

| Model | Description | #Params | Download |
| --- | --- | --- | --- |
| TRELLIS-image-large | Large image-to-3D model | 1.2B | [Download](https://huggingface.co/microsoft/TRELLIS-image-large) |

> **Note:** All VAEs are included in the **TRELLIS-image-large** model repository.

The models are hosted on Hugging Face. You can directly load the models with their repository names in the code:

```python
TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
```

If you prefer loading the model from local storage, you can download the model files from the links above and load the model with the folder path (folder structure should be maintained):

```python
TrellisImageTo3DPipeline.from_pretrained("pretrained_models/TRELLIS-image-large")
```


<!-- Dataset -->
## 📚 Dataset

We provide multiple datasets and their download and preprocessing methods.

### **Objaverse(XL)**

Objaverse(XL) is a large-scale dataset containing various 3D assets. You can download the dataset from the official [Objaverse(XL)](https://objaverse.allenai.org/) website. Put the downloaded dataset into **dataset** directory.

Use the following command for batch rendering the data. You should pre-download [Blender](https://www.blender.org/download/) and change the Blender's path in [render_objaverse.py](dataset_toolkits/render_objaverse.py). Arrange your testing cases in the same directory and change the **down_list** to your testing directory.

```bash
python dataset_toolkits/render_objaverse.py
```

### **ShapeNetCore**

ShapeNetCore is a subset of the full ShapeNet dataset with single clean 3D models and manually verified categories. You can download the dataset from the official [ShapeNetCore](https://shapenet.org/) website. Put the downloaded dataset into **dataset** directory.

Use the following command for batch rendering the data. You should pre-download [Blender](https://www.blender.org/download/) and change the Blender's path in [render_shapenet.py](dataset_toolkits/render_shapenet.py). Arrange your testing cases into a txt file and change the **filter_file** to your testing file.

```bash
python dataset_toolkits/render_shapenet.py
```

Use the following command for generating ground truth for ShapeNetCore. You should pre-download [PartNet](https://drive.google.com/file/d/1NvbGIC-XqZGs9pz6wgFwwEPALR-iR8E0/view) and change the dataset path in [process_shapenet_gt.py](dataset_toolkits/process_shapenet_gt.py). Arrange your testing cases into a txt file and change the **filter_file** to your testing file.

```bash
python dataset_toolkits/process_shapenet_gt.py
```

<!-- Usage -->
## 📖 Usage

### **Objaverse(XL)**

Here is the batch testing file [objaverse_dense](objaverse_cor/objaverse_dense.py) demonstrating how to use NSR to get dense matching between Objaverse(XL) 3D models.

Use the following command to perform dense matching between source and target models.  
First, download the TRELLIS checkpoint, update the checkpoint root directory, and set **test_file** and **test_id** to your source model and directory.  
The encoder weights are included in the TRELLIS checkpoint and can be loaded with:  

```bash
encoder = models.from_pretrained("pretrained_models/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16").eval().cuda()
```

The python script allows you to freely adjust hyperparameters of timesteps and layer. You can set **extract_t** and **extract_l** to **None** to batch test all hyperparameters.

Keypoint matching results can be found in the output directory and visualized into colored .ply files.

```bash
python objaverse_cor/objaverse_dense.py
```

### **ShapeNetCore**

Here is the batch testing file [shapenet_part](shapenet_part/shapenet_part.py) demonstrating how to use NSR to get co-segmentation between ShapeNetCore 3D models.

Use the following command to perform co-segmentation between source and target models. Notice that you should pre-download TRELLIS-checkpoint and change the checkpoint root directory in the file and set your testing source model and directory to **test_file** and **test_id**. Compared to dense matching, you should also provide a source part-segmentation on source vertices to a .json file and set it to **test_label_path**.

The python script allows you to freely adjust hyperparameters of timesteps and layer. You can set **extract_t** and **extract_l** to **None** to batch test all hyperparameters.

Co-segmentation results can be found in the output directory and visualized into colored .ply files.

```bash
python shapenet_part/shapenet_part.py
```

## 🔧 Additional Applications

Once you obtain the dense correspondence results, you can leverage these mappings to create fine-grained correspondences from source to target models. This enables various downstream applications:

- **Keypoint Matching**: Establish precise keypoint correspondences between 3D models
- **Texture Transfer**: Transfer textures between models using dense or part-based mappings
- **Shape Co-segmentation**: Perform semantic segmentation across different 3D shapes
<!-- Citation -->

## 📬 Feedback
If you have any questions, please open an issue or contact **dukeyu@std.uestc.edu.cn**.

## 📜 Citation

If you find this work helpful, please consider citing our paper:

```bibtex
@article{du2025hnsr,
    title   = {Hierarchical Neural Semantic Representation for 3D Semantic Correspondence},
    author  = {Keyu Du and Jingyu Hu and Haipeng Li and Hao Xu and Haibing Huang and Chi-Wing Fu and Shuaicheng Liu},
    journal = {SIGGRAPH ASIA},
    year    = {2025}
}
```

## 🌟 Acknowledgements

- [Trellis](https://github.com/microsoft/TRELLIS)
