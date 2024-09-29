# GSplatLoc: Grounding Keypoint Descriptors into 3D Gaussian Splatting for Improved Visual Localization
[[Paper]](https://arxiv.org/abs/2409.16502) [[Project Page]](https://gsplatloc.github.io/) [[Video]](https://www.youtube.com/watch?v=3UKQQPLlqqg)

**Authors:** Gennady Sidorov, Malik Mohrat, Ksenia Lebedeva, Ruslan Rakhimov, Sergey Kolyubin

![SplatLoc](./assets/pipeline.png)

This repository contains the code for the paper "GSplatLoc: Grounding Keypoint Descriptors into 3D Gaussian Splatting for Improved Visual Localization". 


# Environment setup
Our default, provided install method is based on Conda package and environment management:
<!-- ```
conda env create --file environment.yml
conda activate gsplatloc
``` -->

```shell
conda create --name gsplatloc python=3.10
conda activate gsplatloc
```
PyTorch (Please check your CUDA version, we used 11.8)
```shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Required packages
```shell
pip install -r requirements.txt
```

Submodules

```shell
pip install submodules/diff-gaussian-rasterization # Rasterizer for RGB, n-dim feature, depth
pip install submodules/simple-knn
```

# Data preparation

For main evaluation, we used 7Scenes and Cambridge Landmarks.

Below are the instructions to download and prepare the datasets.

## 7Scenes

You can use the `datasets/setup_7scenes.py` script to download and prepare the data.
We experimented with _Pseudo Ground Truth (PGT)_ camera poses obtained after running SfM on the scenes, as they are more precise than the original D-SLAM poses.

To download and prepare the datasets using the PGT poses:

```python
# Downloads the data to datasets/pgt_7scenes_{chess, fire, ...}
python datasets/setup_7scenes.py --poses pgt
``` 

To complete the dataset preparation, follow these additional steps:

1. Download the SfM models from the [visloc_pseudo_gt_limitations repository](https://github.com/tsattler/visloc_pseudo_gt_limitations/).
2. Extract the downloaded models into the `datasets/` folder.

These SfM model point clouds are used for initializing the 3D Gaussian Splatting (3DGS) process.
```shell
cd datasets

# Downloads sfm models for 7scenes
gdown https://drive.google.com/uc?id=1ATijcGCgK84NKB4Mho4_T-P7x8LSL80m 
unzip 7scenes_reference_models.zip && rm 7scenes_reference_models.zip
``` 


## Cambridge Landmarks

You can download and prepare the Cambridge Landmarks dataset using the script:

```shell
cd datasets

# Downloads the data to datasets/Cambridge_{GreatCourt, KingsCollege, ...}
python datasets/setup_cambridge.py
```

# Training 


```
python train.py -s data/DATASET_NAME -m output/OUTPUT_NAME --iterations 7000
```
<details>
<summary><span style="font-weight: bold;">Command Line Arguments for train.py</span></summary>
  
  #### --source_path / -s
  Path to the source directory containing a COLMAP or Synthetic NeRF data set.
  #### --model_path / -m 
  Path where the trained model should be stored (```output/<random>``` by default).
  #### --images / -i
  Alternative subdirectory for COLMAP images (```images``` by default).
  #### --eval
  Add this flag to use a MipNeRF360-style training/test split for evaluation.
  #### --resolution / -r
  Specifies resolution of the loaded images before training. If provided ```1, 2, 4``` or ```8```, uses original, 1/2, 1/4 or 1/8 resolution, respectively. If proveided ```0```, use GT feature map's resolution. For all other values, rescales the width to the given number while maintaining image aspect. If proveided ```-2```, use the customized resolution (```utils/camera_utils.py L31```). **If not set and input image width exceeds 1.6K pixels, inputs are automatically rescaled to this target.**
  #### --speedup
  Optional speed-up module for reduced feature dimention initialization.
  #### --data_device
  Specifies where to put the source image data, ```cuda``` by default, recommended to use ```cpu``` if training on large/high-resolution dataset, will reduce VRAM consumption, but slightly slow down training. Thanks to [HrsPythonix](https://github.com/HrsPythonix).
  #### --white_background / -w
  Add this flag to use white background instead of black (default), e.g., for evaluation of NeRF Synthetic dataset.
  #### --sh_degree
  Order of spherical harmonics to be used (no larger than 3). ```3``` by default.
  #### --convert_SHs_python
  Flag to make pipeline compute forward and backward of SHs with PyTorch instead of ours.
  #### --convert_cov3D_python
  Flag to make pipeline compute forward and backward of the 3D covariance with PyTorch instead of ours.
  #### --debug
  Enables debug mode if you experience erros. If the rasterizer fails, a ```dump``` file is created that you may forward to us in an issue so we can take a look.
  #### --debug_from
  Debugging is **slow**. You may specify an iteration (starting from 0) after which the above debugging becomes active.
  #### --iterations
  Number of total iterations to train for, ```30_000``` by default.
  #### --ip
  IP to start GUI server on, ```127.0.0.1``` by default.
  #### --port 
  Port to use for GUI server, ```6009``` by default.
  #### --test_iterations
  Space-separated iterations at which the training script computes L1 and PSNR over test set, ```7000 30000``` by default.
  #### --save_iterations
  Space-separated iterations at which the training script saves the Gaussian model, ```7000 30000 <iterations>``` by default.
  #### --checkpoint_iterations
  Space-separated iterations at which to store a checkpoint for continuing later, saved in the model directory.
  #### --start_checkpoint
  Path to a saved checkpoint to continue training from.
  #### --quiet 
  Flag to omit any text written to standard out pipe. 
  #### --feature_lr
  Spherical harmonics features learning rate, ```0.0025``` by default.
  #### --opacity_lr
  Opacity learning rate, ```0.05``` by default.
  #### --scaling_lr
  Scaling learning rate, ```0.005``` by default.
  #### --rotation_lr
  Rotation learning rate, ```0.001``` by default.
  #### --position_lr_max_steps
  Number of steps (from 0) where position learning rate goes from ```initial``` to ```final```. ```30_000``` by default.
  #### --position_lr_init
  Initial 3D position learning rate, ```0.00016``` by default.
  #### --position_lr_final
  Final 3D position learning rate, ```0.0000016``` by default.
  #### --position_lr_delay_mult
  Position learning rate multiplier (cf. Plenoxels), ```0.01``` by default. 
  #### --densify_from_iter
  Iteration where densification starts, ```500``` by default. 
  #### --densify_until_iter
  Iteration where densification stops, ```15_000``` by default.
  #### --densify_grad_threshold
  Limit that decides if points should be densified based on 2D position gradient, ```0.0002``` by default.
  #### --densification_interval
  How frequently to densify, ```100``` (every 100 iterations) by default.
  #### --opacity_reset_interval
  How frequently to reset opacity, ```3_000``` by default. 
  #### --lambda_dssim
  Influence of SSIM on total loss from 0 to 1, ```0.2``` by default. 
  #### --percent_dense
  Percentage of scene extent (0--1) a point must exceed to be forcibly densified, ```0.01``` by default.

</details>


In this work, we didn't use the feature-3dgs speed-up module.
The `diff-gaussian-rasterization` module is designed for 64-dimensional XFeat descriptors, but it can accommodate any 64-dimensional feature vector.

If you wish to use a different feature dimension from a different encoder, you can modify the `NUM_SEMANTIC_CHANNELS` parameter in the `config.h` file within the cuda-rasterizer directory and rebuild the module.


# Localization

The main localization pipeline is implemented in `loc_inference.py`.
Here you can find the **pose prior estimation** and **pose refinement** modules.

The basic usage is as follows:

```Python
# Specify the path to the trained model
# Additional parameters can be set as needed (see below for options)
python loc_inference.py -m output/OUTPUT_NAME
```
The pipeline parameters is also can be adjusted.

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for loc_inference.py</span></summary>
  
  #### --model_path / -m 
  Path to the trained model directory you want to create renderings for.
  #### --top_k
  Number of top reliable keypoints from XFeat.
  #### --ransac_iters
  Number of PnP-RANSAC iterations.
  #### --warp_lr
  Learning rate for pose refinement.
  #### --warp_iters
  Number of warp iterations.

</details>


# Acknowledgements

This project builds upon and extends the work of several open-source projects:

- [Feature 3DGS](https://github.com/ShijieZhou-UCLA/feature-3dgs)
- [XFeat](https://github.com/verlab/accelerated_features)
- [PNeRFLoc](https://github.com/BoMingZhao/PNeRFLoc)
- [ACE](https://github.com/nianticlabs/ace)

We are deeply grateful to the authors and contributors of these projects for making their code available to the research community.


<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@misc{sidorov2024gsplatlocgroundingkeypointdescriptors,
      title={GSplatLoc: Grounding Keypoint Descriptors into 3D Gaussian Splatting for Improved Visual Localization}, 
      author={Gennady Sidorov and Malik Mohrat and Ksenia Lebedeva and Ruslan Rakhimov and Sergey Kolyubin},
      year={2024},
      eprint={2409.16502},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.16502}, 
}</code></pre>
  </div>
</section>



