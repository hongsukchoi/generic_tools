import numpy as np

# generate the 360 rotating rendering view for the target object, when there is only one camera. If there is no world coordinate, you can pass identity matrix for world2cam
def generate_rotating_rendering_path(world2cam, object_center, num_render_views=60):
    """
    world2cam: (4,4) numpy matrix, transformation matrix that transforms a 3D point in the world coordinate to the camera coordinate
    object_center: 3D location of object center in the camera coordinate 
    num_render_views: number of breakdown for 360 degree
    
    """

    lower_row = np.array([[0., 0., 0., 1.]])

    object_center_to_camera_origin = -object_center

    # output; list of transformation matrices that transform a 3D point in the world coordinate to a (rotating) camera coordinate
    render_w2c = []
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

        # get the transformation of coordiante-axis from the original camera to the new camera == transformation matrix that transforms a 3D point in the new camera coordinate to the original camera coordinate
        # 원래 카메라에서 새로운 카메라로의 좌표축변환행렬
        R_newcam2origcam = np.stack([x_axis, y_axis, z_axis], axis=1)
        newcam2origcam = np.concatenate(
            [R_newcam2origcam, new_camera_origin[:, None]], axis=1)
        newcam2origcam = np.concatenate([newcam2origcam, lower_row], axis=0)
        
        # transformation matrix that transforms a 3D point in the original camera coordinate to the new camera coordinate
        origcam2newcam = np.linalg.inv(newcam2origcam)

        # transformation matrix that transforms a 3D point in the world camera coordinate to the new camera coordinate
        world2newcam = origcam2newcam @ world2cam

        render_w2c.append(world2newcam)

    return render_w2c
