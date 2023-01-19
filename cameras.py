import numpy as np


def look_at_rotation(camera_centers: np.ndarray, at: np.ndarray, up: np.ndarray) -> np.ndarray:
    """
    camera_centers: (N, 3) numpy matrix, locations of cameras in the world coordinate
    at: (1, 3) numpy matrix, a location where cameras should look at
    up: (1, 3) numpy matrix, an up vector that the cameras should have

    // Returns //
    world2cam_R: (N, 3, 3) numpy matrix, transforms a 3D point in the world coordinate to the camera coordinate

    """
    z_axes = at - camera_centers
    z_axes = z_axes / np.linalg.norm(z_axes, axis=1, keepdims=True)
    x_axes = np.cross(up, z_axes, axis=1)
    x_axes = x_axes / np.linalg.norm(x_axes, axis=1, keepdims=True)
    y_axes = np.cross(z_axes, x_axes, axis=1)
    y_axes = y_axes / np.linalg.norm(y_axes, axis=1, keepdims=True)

    is_close = np.isclose(x_axes, np.array(
        0.0), atol=5e-3).all(axis=1, keepdims=True)
    if is_close.any():
        replacement = np.cross(y_axes, z_axes, axis=1)
        replacement = replacement / \
            np.linalg.norm(replacement, axis=1, keepdims=True)
        x_axes = np.where(is_close, replacement, x_axes)

    # (N, 3, 3)
    # convert x and y axes to match the conventional camera
    R = np.concatenate(
        (-x_axes[:, :, None], -y_axes[:, :, None], z_axes[:, :, None]), axis=2)

    world2cam_R = R.transpose(0, 2, 1)

    return world2cam_R

def look_at_view_transform(camera_centers: np.ndarray, at: np.ndarray, up: np.ndarray) -> np.ndarray:
    """
    camera_centers: (N, 3) numpy matrix, locations of cameras in the world coordinate
    at: (1, 3) numpy matrix, a location where cameras should look at
    up: (1, 3) numpy matrix, an up vector that the cameras should have

    // Returns //
    R: (N, 3, 3) numpy matrix, world2cam
    t: (N, 3) numpy matrix, world2cam
    """

    R = look_at_rotation(camera_centers, at, up)
    t = - np.matmul(R, camera_centers[:, :, None])[:, :, 0]  
    
    return R, t