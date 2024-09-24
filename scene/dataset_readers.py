
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text, read_points3D_nvm
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from torchvision.transforms import PILToTensor
from encoders.XFeat.modules.xfeat import XFeat

import torch

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    semantic_feature: torch.tensor 


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    semantic_feature_dim: int 

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

@torch.inference_mode()
def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    model = XFeat().cuda()
    
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE" or intr.model=="SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        ### elif intr.model=="PINHOLE":
        elif intr.model=="PINHOLE" or intr.model=="OPENCV":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"


        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        try:
            image = Image.open(image_path) 
        except:
            print(f"Error opening image: {image_path}")
            continue
      
        tensor_image = PILToTensor()(image)[None].float()
      
        semantic_feature = model.get_descriptors(tensor_image)[0]

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1],
                            semantic_feature=semantic_feature)
        
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, foundation_model, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    
    reading_dir = "images" if images == None else images

 
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, 
                                           images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    semantic_feature_dim = cam_infos[0].semantic_feature.shape[0]

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 2] # avoid 1st to be test view
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 2] 
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
            
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           semantic_feature_dim=semantic_feature_dim) 
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, semantic_feature_folder, extension=".png"): 
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            semantic_feature_path = os.path.join(semantic_feature_folder, image_name) + '_fmap_CxHxW.pt' 
            semantic_feature_name = os.path.basename(semantic_feature_path).split(".")[0]
            semantic_feature = torch.load(semantic_feature_path)
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1],
                              semantic_feature=semantic_feature,
                              semantic_feature_path=semantic_feature_path,
                              semantic_feature_name=semantic_feature_name)) 
            
    return cam_infos


def readSplit_cams_params(intrinsic_folder, extrinsic_folder):

    intrinsic_files = sorted(os.listdir(intrinsic_folder))
    extrinsic_files = sorted(os.listdir(extrinsic_folder))

    # Read intrinsics

    with open(f"{intrinsic_folder}/{intrinsic_files[0]}", "r") as fid:
        K = float(fid.readline())


    w2cs = []

    # Read extrincsics

    for file in extrinsic_files:

        with open(f"{extrinsic_folder}/{file}", "r") as fid:

            c2w = []

            while True:
                line = fid.readline().rstrip()

                if not line:
                    break
                
                c2w+=line.split(' ')

        c2w = np.array([float(x) for x in c2w]).reshape((4,4))
        w2c = np.linalg.inv(c2w)
        # RTs
        w2cs.append(w2c)

    return K, w2cs

@torch.inference_mode()
def readSplitInfo(path, pcd = None):
    
    train_images_folder = os.path.join(path, "train/rgb")
    train_extrinsic_folder = os.path.join(path, "train/poses")
    train_intrinsic_folder = os.path.join(path, "train/calibration")
    
    test_images_folder = os.path.join(path, "test/rgb")
    test_extrinsic_folder = os.path.join(path, "test/poses")
    test_intrinsic_folder = os.path.join(path, "test/calibration")

    ply_path = os.path.join(path, "out.ply")

    scene_name = path.split("_")[-1]    

    if '7scenes' in path:
        sfm_path = os.path.join(f"{path.split(scene_name)[0].replace('pgt_', '')}reference_models", scene_name, "old_gt_refined")

    elif 'Cambridge' in path:
        sfm_path = path
    
    else:
        raise ValueError(f"Unknown dataset: {path}")

    print(sfm_path)

    train_views = sorted(os.listdir(train_images_folder))
    test_views = sorted(os.listdir(test_images_folder))

    train_cam_infos_unsorted = []
    test_cam_infos_unsorted = []

    K_train, w2cs_train = readSplit_cams_params(train_intrinsic_folder, train_extrinsic_folder)
    K_test, w2cs_test = readSplit_cams_params(test_intrinsic_folder, test_extrinsic_folder)

    width, height = Image.open(f"{train_images_folder}/{train_views[0]}").size
    
    model = XFeat().cuda()

    for i, view in enumerate(train_views):
        sys.stdout.write('\r')
        sys.stdout.write(f"Reading {i+1} train / {len(train_views)} camera")
        sys.stdout.flush()

        w2c_sample = w2cs_train[i]
        
        R = w2c_sample[:3,:3].T  # R is stored transposed due to 'glm' in CUDA code
        T = w2c_sample[:3, 3]
        
        focal_length_x = K_train
        FovY = focal2fov(focal_length_x, height)
        FovX = focal2fov(focal_length_x, width)

        image_path = os.path.join(train_images_folder, view)
        image_name = os.path.basename(image_path).split(".png")[0]
        image = Image.open(image_path)
      
        tensor_image = PILToTensor()(image)[None].float()
        semantic_feature = model.get_descriptors(tensor_image)[0]
   
        cam_info = CameraInfo(uid=i, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1],
                            semantic_feature=semantic_feature)
        
        train_cam_infos_unsorted.append(cam_info)
    

    for i, view in enumerate(test_views):
        sys.stdout.write('\r')
        sys.stdout.write(f"Reading {i+1} test / {len(test_views)} camera")
        sys.stdout.flush()

        w2c_sample = w2cs_test[i]

        R = w2c_sample[:3,:3].T  # R is stored transposed due to 'glm' in CUDA code
        T = w2c_sample[:3, 3]

        focal_length_x = K_test
        FovY = focal2fov(focal_length_x, height)
        FovX = focal2fov(focal_length_x, width)

        image_path = os.path.join(test_images_folder, view)
        image_name = os.path.basename(image_path).split(".png")[0]
        image = Image.open(image_path)
      
        tensor_image = PILToTensor()(image)[None].float()
        semantic_feature = model.get_descriptors(tensor_image)[0]

        cam_info = CameraInfo(uid=i, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1],
                            semantic_feature=semantic_feature)
        test_cam_infos_unsorted.append(cam_info)
    
    train_cam_infos = sorted(train_cam_infos_unsorted, key = lambda x : x.image_name)
    test_cam_infos = sorted(test_cam_infos_unsorted, key = lambda x : x.image_name)

    print(f"\nTotal cams: {len(train_cam_infos)+len(test_cam_infos)}")

    nerf_normalization = getNerfppNorm(train_cam_infos)

    if 'Cambridge' in path:
        nvm_path = os.path.join(sfm_path, "reconstruction.nvm")

        if not os.path.exists(ply_path):
            print("Converting reconstruction.nvm to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb= read_points3D_nvm(nvm_path)
            storePly(ply_path, xyz, rgb)
        except:
            print("Error reading reconstruction.nvm file. Please ensure it exists and is in the correct format.")

    else:

        bin_path = os.path.join(sfm_path, "points3D.bin")

        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
            storePly(ply_path, xyz, rgb)
        except:
            print("Error reading reconstruction.nvm file. Please ensure it exists and is in the correct format.")
    try:
        pcd = fetchPly(ply_path)
    except:
        print("Error reading .ply file. Using default point cloud.")
        pcd = None
    
    semantic_feature_dim = train_cam_infos[0].semantic_feature.shape[0]

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           semantic_feature_dim=semantic_feature_dim)

    return scene_info



def readNerfSyntheticInfo(path, foundation_model, white_background, eval, extension=".png"): 
    if foundation_model =='sam':
        semantic_feature_dir = "sam_embeddings" 
    elif foundation_model =='lseg':
        semantic_feature_dir = "rgb_feature_langseg" 

    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, semantic_feature_folder=os.path.join(path, semantic_feature_dir)) 
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, semantic_feature_folder=os.path.join(path, semantic_feature_dir)) 
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    semantic_feature_dim = train_cam_infos[0].semantic_feature.shape[0] 
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           semantic_feature_dim=semantic_feature_dim) 
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Split" : readSplitInfo
}