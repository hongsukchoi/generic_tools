import numpy as np
from typing import List
from coordinate import fit_circle_in_3d, _disambiguate_normal
from cameras import look_at_view_transform

# Generate the 360 rotating rendering view for the target object, when there is only one camera. If there is no world coordinate, you can pass identity matrix for world2cam
def generate_rotating_rendering_path(world2cam: np.ndarray, object_center: np.ndarray, num_render_views: int=60) -> List[np.ndarray]:
    """
    world2cam: (4,4) numpy matrix, transformation matrix that transforms a 3D point in the world coordinate to the camera coordinate
    object_center: (3,) numpy vector, 3D location of object center in the camera coordinate 
    num_render_views: scalar, number of breakdown for 360 degree
    
    // Return // 
    rotating_world2cam_list: list of (4,4) transformation matrices that transform a 3D point in the world coordinate to a (rotating) camera coordinate
    """

    lower_row = np.array([[0., 0., 0., 1.]])

    object_center_to_camera_origin = -object_center

    rotating_world2cam_list = []
    for theta in np.linspace(0., 2 * np.pi, num_render_views + 1)[:-1]:
        # transformation from original camera to a new camera 
        # theta = - np.pi / 6  # 30 degree
        sin, cos = np.sin(theta), np.cos(theta)
        augR = np.eye(3)
        augR[0, 0], augR[2, 2] = cos, cos
        augR[0, 2], augR[2, 0] = sin, -sin

        # rotate around the camera's y-axis, around the object center
        new_camera_origin = augR @ object_center_to_camera_origin + object_center

        # the new camera's z-axis; it should point to the object 
        z_axis = object_center - new_camera_origin
        z_axis = z_axis / np.linalg.norm(z_axis)
        # we are trying to rotate around the y-axis' so y-axis remains the same
        y_axis = np.array([0., 1., 0.])
        # get the x-axis of the new camera
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        # get z_axis again to make a valid rotation matrix
        # convert to correct rotation matrix
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)

        # get the (4,4) transformation of coordiante-axis from the original camera to the new camera == transformation matrix that transforms a 3D point in the new camera coordinate to the original camera coordinate
        # 원래 카메라에서 새로운 카메라로의 좌표축변환행렬
        R_newcam2origcam = np.stack([x_axis, y_axis, z_axis], axis=1)
        newcam2origcam = np.concatenate(
            [R_newcam2origcam, new_camera_origin[:, None]], axis=1)
        newcam2origcam = np.concatenate([newcam2origcam, lower_row], axis=0)
        
        # transformation matrix that transforms a 3D point in the original camera coordinate to the new camera coordinate
        origcam2newcam = np.linalg.inv(newcam2origcam)

        # transformation matrix that transforms a 3D point in the world camera coordinate to the new camera coordinate
        world2newcam = origcam2newcam @ world2cam

        rotating_world2cam_list.append(world2newcam)

    return rotating_world2cam_list


# Generate the 360 rotating view, but evenly distributed, from the unevenly rotating camera views
def generate_rendering_path_from_multiviews(camera_centers: np.ndarray, world_object_center: np.ndarray, up: np.ndarray = np.array([0., -1., 0.], dtype=np.float32), num_render_views: int= 60, trajectory_scale: float = 1.1) -> List[np.ndarray]:
    """
    camera_centers: (N, 3) numpy matrix, 3D locations of cameras in the world coordinate
    world_object_center: (3,) numpy vector, 3D location of object center in the world coordinate
    up: (3,) numpy vector, up vector of the world coordinate
    
    // Returns //
    Rts: (num_rendering_view, 4, 4) numpy matrix, transformation matrix that transforms a 3D point in the world coordinate to the rotating camera coordinates
    """

    angles = np.linspace(0, 2.0 * np.pi, num_render_views).astype(np.float32)

    # rendering_camera_centers: (num_rendering_views, 3), normal: (3,)
    rendering_camera_centers, normal = fit_circle_in_3d(camera_centers, angles=angles) 

    # align the normal to up vector of the world corodinate
    up = _disambiguate_normal(normal, up)
    
    # scale the distance between the rotating cameras and the object center in the world coordinate
    traj = rendering_camera_centers
    _t_mu = traj.mean(axis=0, keepdims=True)
    traj = (traj - _t_mu) * trajectory_scale + _t_mu

    # point all cameras towards the center of the scene
    Rs, ts = look_at_view_transform(
        traj,
        at=world_object_center[None, :],  # (1, 3)
        up=up[None, :],  # (1, 3)
    )

    Rts = np.concatenate([Rs, ts[:, :, None]], axis=2)  # (num_rendering_views, 3, 4)
    Rts = np.concatenate([Rts, np.array(
        [[[0., 0., 0., 1.]]], dtype=np.float32).repeat(Rts.shape[0], axis=0)], axis=1)


    return Rts