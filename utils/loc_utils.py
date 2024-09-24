import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F


def calculate_pose_errors(R_gt, t_gt, R_est, t_est):
    # Calculate rotation error
    rotError = np.matmul(R_est.T, R_gt)
    rotError = cv2.Rodrigues(rotError)[0]
    rotError = np.linalg.norm(rotError) * 180 / np.pi

    # Calculate translation error
    transError = np.linalg.norm(t_gt - t_est.squeeze(1)) * 100  # Convert to cm

    return rotError, transError

def log_errors(model_path, name, rotation_errors, translation_errors, inplace_text):
    
    total_frames = len(rotation_errors)
    # Remove NaN values from rotation_errors and translation_errors
    rotation_errors = [err for err in rotation_errors if not np.isnan(err)]
    translation_errors = [err for err in translation_errors if not np.isnan(err)]

    # Ensure both lists have the same length after NaN removal
    min_length = min(len(rotation_errors), len(translation_errors))
    rotation_errors = rotation_errors[:min_length]
    translation_errors = translation_errors[:min_length]

    # Update total_frames after NaN removal
    total_frames = len(rotation_errors)
    median_rErr = np.median(rotation_errors)
    median_tErr = np.median(translation_errors)

    # Compute accuracy percentages
    pct10_5 = sum(r <= 5 and t <= 10 for r, t in zip(rotation_errors, translation_errors)) / total_frames * 100
    pct5 = sum(r <= 5 and t <= 5 for r, t in zip(rotation_errors, translation_errors)) / total_frames * 100
    pct2 = sum(r <= 2 and t <= 2 for r, t in zip(rotation_errors, translation_errors)) / total_frames * 100
    pct1 = sum(r <= 1 and t <= 1 for r, t in zip(rotation_errors, translation_errors)) / total_frames * 100

    print('Accuracy:')
    print(f'\t10cm/5deg: {pct10_5:.1f}%')
    print(f'\t5cm/5deg: {pct5:.1f}%')
    print(f'\t2cm/2deg: {pct2:.1f}%')
    print(f'\t1cm/1deg: {pct1:.1f}%')
    print(f'\tmedian_rErr: {median_rErr:.3f} deg')
    print(f'\tmedian_tErr: {median_tErr:.3f} cm')

    # Log median errors to separate files
    log_dir = os.path.join(model_path, 'error_logs')
    os.makedirs(log_dir, exist_ok=True)
    
    with open(os.path.join(log_dir, f'median_error_{name}_{inplace_text}_end.txt'), 'w') as f:
            
        f.write('Accuracy:\n')
        f.write(f'\t10cm/5deg: {pct10_5:.1f}%\n')
        f.write(f'\t5cm/5deg: {pct5:.1f}%\n')
        f.write(f'\t2cm/2deg: {pct2:.1f}%\n')
        f.write(f'\t1cm/1deg: {pct1:.1f}%\n')
        f.write(f'Median translation error: {median_tErr:.6f} cm\n')
        f.write(f'Median rotation error: {median_rErr:.6f} dg\n')


def log_errors_iters(model_path, name, rotation_errors, translation_errors, inplace_text):
    
    # Remove NaN values from rotation_errors and translation_errors
    rotation_errors = {iter_num: [err for err in errors if not np.isnan(err)] for iter_num, errors in rotation_errors.items()}
    translation_errors = {iter_num: [err for err in errors if not np.isnan(err)] for iter_num, errors in translation_errors.items()}

    log_dir = os.path.join(model_path, 'error_logs')
    os.makedirs(log_dir, exist_ok=True)

    with open(os.path.join(log_dir, f'median_error_{name}_{inplace_text}_iters_end.txt'), 'w') as f:
        for iter_num in sorted(rotation_errors.keys()):
            median_rErr = np.median(rotation_errors[iter_num])
            median_tErr = np.median(translation_errors[iter_num])
            f.write(f'{iter_num} iter | t_err: {median_tErr:.6f} cm, r_err: {median_rErr:.6f} deg\n')



def find_2d3d_correspondences(keypoints, image_features, gaussian_pcd, gaussian_feat, chunk_size=10000):
    device = image_features.device
    f_N, feat_dim = image_features.shape
    P_N = gaussian_feat.shape[0]
    
    # Normalize features for faster cosine similarity computation
    image_features = F.normalize(image_features, p=2, dim=1)
    gaussian_feat = F.normalize(gaussian_feat, p=2, dim=1)
    
    max_similarity = torch.full((f_N,), -float('inf'), device=device)
    max_indices = torch.zeros(f_N, dtype=torch.long, device=device)
    
    for part in range(0, P_N, chunk_size):
        chunk = gaussian_feat[part:part + chunk_size]
        # Use matrix multiplication for faster similarity computation
        similarity = torch.mm(image_features, chunk.t())
        
        chunk_max, chunk_indices = similarity.max(dim=1)
        update_mask = chunk_max > max_similarity
        max_similarity[update_mask] = chunk_max[update_mask]
        max_indices[update_mask] = chunk_indices[update_mask] + part

    point_vis = gaussian_pcd[max_indices].cpu().numpy().astype(np.float64)
    keypoints_matched = keypoints[..., :2].cpu().numpy().astype(np.float64)
    
    return point_vis, keypoints_matched