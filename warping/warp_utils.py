import numpy as np
import torch



def get_cam_tensor(c2w, device='cuda'):
    
    R = c2w[:3, :3]
    quad = rotmat2qvec_tensor(R).view([4])
    t = c2w[:3, 3].view([3])

    camera_tensor = torch.cat([quad, t], dim=0).detach()
    return camera_tensor


def qvec2rotmat_tensor(qvec):
    return torch.nn.functional.normalize(
        torch.stack([
            1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
            2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
            2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
            1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
            2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
            2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
            1 - 2 * qvec[1]**2 - 2 * qvec[2]**2
        ], dim=0).view(3, 3), dim=-1
    )

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def rotmat2qvec_tensor(R):
    K = torch.empty(4, 4, device=R.device)
    K[0, 0] = R[0, 0] - R[1, 1] - R[2, 2]
    K[0, 1] = R[1, 0] + R[0, 1]
    K[0, 2] = R[2, 0] + R[0, 2]
    K[0, 3] = R[2, 1] - R[1, 2]
    K[1, 0] = R[1, 0] + R[0, 1]
    K[1, 1] = R[1, 1] - R[0, 0] - R[2, 2]
    K[1, 2] = R[2, 1] + R[1, 2]
    K[1, 3] = R[0, 2] - R[2, 0]
    K[2, 0] = R[2, 0] + R[0, 2]
    K[2, 1] = R[2, 1] + R[1, 2]
    K[2, 2] = R[2, 2] - R[0, 0] - R[1, 1]
    K[2, 3] = R[1, 0] - R[0, 1]
    K[3, 0] = R[2, 1] - R[1, 2]
    K[3, 1] = R[0, 2] - R[2, 0]
    K[3, 2] = R[1, 0] - R[0, 1]
    K[3, 3] = R[0, 0] + R[1, 1] + R[2, 2]
    K /= 3.0

    eigvals, eigvecs = torch.linalg.eigh(K)
    qvec = eigvecs[:, torch.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec[[3, 0, 1, 2]]

def from_cam_tensor_to_c2w(cam_tensor):
    R = qvec2rotmat_tensor(cam_tensor[:4])
    t = cam_tensor[4:].unsqueeze(1)
    
    return torch.cat([
        torch.cat([R, t], dim=1),
        torch.tensor([[0., 0., 0., 1.]], device=cam_tensor.device)
    ])