import torch
import torch.nn.functional as F

from kornia.geometry.depth import warp_frame_depth
from warping.warp_utils import *


def differentiable_warp(image, depth, warp, K):
    B, _, H, W = image.shape
    
    # Create pixel coordinates
    y, x = torch.meshgrid(torch.arange(H, device=image.device), torch.arange(W, device=image.device))
    pixels = torch.stack((x, y, torch.ones_like(x)), dim=-1).float()  # (H, W, 3)
    pixels = pixels.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 3)
    
    # Unproject to 3D
    depth = depth.permute(0, 2, 3, 1)  # (B, H, W, 1)
    cam_points = (torch.inverse(K).unsqueeze(1) @ pixels.unsqueeze(-1)).squeeze(-1)  # (B, H, W, 3)
    world_points = depth * cam_points  # (B, H, W, 3)
    
    # Transform points
    world_points = world_points.view(B, -1, 3).transpose(1, 2)  # (B, 3, H*W)
    world_points = torch.cat([world_points, torch.ones_like(world_points[:, :1])], dim=1)  # (B, 4, H*W)
    cam_points = warp @ world_points  # (B, 4, H*W)
    cam_points = cam_points[:, :3] / cam_points[:, 3:4]  # (B, 3, H*W)
    
    # Project to 2D
    proj_points = K @ cam_points  # (B, 3, H*W)
    proj_points = proj_points[:, :2] / proj_points[:, 2:3]  # (B, 2, H*W)
    proj_points = proj_points.view(B, 2, H, W)  # (B, 2, H, W)
    
    # Normalize coordinates
    proj_points[:, 0] = (proj_points[:, 0] / (W - 1)) * 2 - 1
    proj_points[:, 1] = (proj_points[:, 1] / (H - 1)) * 2 - 1
    proj_points = proj_points.permute(0, 2, 3, 1)  # (B, H, W, 2)
    
    # Sample from image
    warped_image = F.grid_sample(image, proj_points, mode='bilinear', padding_mode='zeros', align_corners=True)
    
    return warped_image

# Use in compute_warping_loss
def compute_warping_loss(vr, qr, quat_opt, t_opt, pose, K, depth):

    warp = pose @ from_cam_tensor_to_w2c(torch.cat([quat_opt, t_opt], dim=0)).inverse()
    
    warped_image = differentiable_warp(vr.unsqueeze(0), depth.unsqueeze(0), warp.unsqueeze(0), K.unsqueeze(0))
    loss = F.mse_loss(warped_image, qr.unsqueeze(0))

    return loss


def compute_warping_loss_kornia(vr, qr, quat_opt, t_opt, pose, K, depth):

    warp = pose @ from_cam_tensor_to_w2c(torch.cat([quat_opt, t_opt], dim=0)).inverse()
    
    warped_image = warp_frame_depth(image_src=vr.unsqueeze(0),
                                    depth_dst=depth.unsqueeze(0),
                                    src_trans_dst=warp.unsqueeze(0),
                                    camera_matrix=K.unsqueeze(0),
                                    normalize_points=False)
        
    loss = F.mse_loss(warped_image, qr.unsqueeze(0))

    return loss
