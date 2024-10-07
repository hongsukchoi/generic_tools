import os
from pathlib import Path


import open3d as o3d
import numpy as np
import cv2
import viser.transforms as vtf

from scipy.spatial.transform import Rotation as R
from projectaria_tools.core import data_provider, calibration, mps


# Image size
image_size = (512, 512)
focal_length = 150


def get_undistorted_rotated_image(stream_id, frame_idx, src_calib, dst_calib):
    raw_image = provider.get_image_data_by_index(stream_id, frame_idx)[0].to_numpy_array()
    undistorted_image = calibration.distort_by_calibration(raw_image, dst_calib, src_calib)

    # Rotated image by CW90 degrees
    rotated_image = np.rot90(undistorted_image, k=3)

    # Get rotated image calibration
    pinhole_cw90 = calibration.rotate_camera_calib_cw90deg(dst_calib)

    return rotated_image, pinhole_cw90

""" hard coding """ 
# CameraCalibration(label: camera-slam-left, model name: Linear, principal point: [255.5, 255.5], focal length: [150, 150], projection params: [150, 150, 255.5, 255.5], image size (w,h): [512, 512], T_Device_Camera:(translation:[-1.73472e-18, -5.55112e-17, -1.04083e-17], quaternion(x,y,z,w):[0, 0, -0.707107, 0.707107]), serialNumber:LinearCameraCalibration)
# CameraCalibration(label: camera-slam-right, model name: Linear, principal point: [255.5, 255.5], focal length: [150, 150], projection params: [150, 150, 255.5, 255.5], image size (w,h): [512, 512], T_Device_Camera:(translation:[0.00648219, -0.111717, -0.087507], quaternion(x,y,z,w):[0.433827, 0.438017, -0.529742, 0.582503]), serialNumber:LinearCameraCalibration)

# Intrinsic parameters for both cameras (focal lengths and principal points)
K_left = np.array([[focal_length, 0, 255.5],
                [0, focal_length, 255.5],
                [0, 0, 1]])

K_right = np.array([[focal_length, 0, 255.5],
                    [0, focal_length, 255.5],
                    [0, 0, 1]])

# Distortion coefficients are set to zero (undistorted images)
D_left = np.zeros(5)
D_right = np.zeros(5)

# Translation vectors (extrinsics)
T_left = np.array([-1.73472e-18, -5.55112e-17, -1.04083e-17])  # Left camera translation (basically zero)
T_right = np.array([0.00648219, -0.111717, -0.087507])  # Right camera translation

# Quaternion for left camera (no significant rotation)
q_left = np.array([0, 0, -0.707107, 0.707107])
# Quaternion for right camera
q_right = np.array([0.433827, 0.438017, -0.529742, 0.582503])

# Convert quaternions to rotation matrices
R_left = R.from_quat(q_left).as_matrix()
R_right = R.from_quat(q_right).as_matrix()

# # Compute the relative rotation (R) and translation (T) between the two cameras
# R_rel = R_right @ R_left.T  # Relative rotation between the cameras
# T_rel = T_right - T_left    # Relative translation (T_right already in left camera frame)

# transformation matrix that transforms a point in the left camera frame to the right camera frame
# https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga617b1685d4059c6040827800e72ad2b6
# https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga9d2539c1ebcda647487a616bdf0fc716
# I guess this is left multiplication? x_r = R_rel @ x_l + T_rel; Yes. 

# frame transformation
# R_left: camera orientation
# T_left: camera origin in world space

# point transformation / left_cam2world
# R_left_new = R_left.T
# T_left_new = - R_left.T @ T_left


# world2left_cam = np.eye(4)
# world2left_cam[:3, :3] = R_left[:3, :3]
# world2left_cam[:3, 3] = T_left[:3]


# world2right_cam = np.eye(4)
# world2right_cam[:3, :3] = R_right[:3, :3]
# world2right_cam[:3, 3] = T_right[:3]

# left_cam2right_cam = np.linalg.inv(world2left_cam) @ world2right_cam

# left_cam2right_cam = np.linalg.inv(left_cam2right_cam)


left_cam2world = np.eye(4)
left_cam2world[:3, :3] = R_left[:3, :3]
left_cam2world[:3, 3] = T_left[:3]


right_cam2world = np.eye(4)
right_cam2world[:3, :3] = R_right[:3, :3]
right_cam2world[:3, 3] = T_right[:3]

left_cam2right_cam = np.linalg.inv(right_cam2world) @ left_cam2world

R_rel = left_cam2right_cam[:3, :3]
T_rel = left_cam2right_cam[:3, 3]


R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K_left, D_left, K_right, D_right, image_size, R_rel, T_rel)

""" hard coding """ 
vrsfile = 'Coke.vrs'
provider = data_provider.create_vrs_data_provider(vrsfile)


# Get calibration data for fisheye camera
# Left
left_camera_label = "camera-slam-left" #

left_stream_id = provider.get_stream_id_from_label(left_camera_label)

left_calib = provider.get_device_calibration().get_camera_calib(left_camera_label)
left_pinhole = calibration.get_linear_camera_calibration(512, 512, focal_length, left_camera_label,
left_calib.get_transform_device_camera())


# Right
right_camera_label = "camera-slam-right" #

right_stream_id = provider.get_stream_id_from_label(right_camera_label)

right_calib = provider.get_device_calibration().get_camera_calib(right_camera_label)
right_pinhole = calibration.get_linear_camera_calibration(512, 512, focal_length, right_camera_label,
right_calib.get_transform_device_camera())



save_dir = './rectified_overlap' # './raw_images'
right_save_dir = os.path.join(save_dir, 'coke_slam_right')
Path(right_save_dir).mkdir(parents=True, exist_ok=True)
left_save_dir = os.path.join(save_dir, 'coke_slam_left')
Path(left_save_dir).mkdir(parents=True, exist_ok=True)

# frame_idx = 0
# while True:
#     try:
#         raw_right_image = provider.get_image_data_by_index(right_stream_id, frame_idx)[0].to_numpy_array()
#         raw_left_image = provider.get_image_data_by_index(left_stream_id, frame_idx)[0].to_numpy_array()
#     except:
#         break

#     cv2.imwrite(os.path.join(right_save_dir, f'{frame_idx:04d}.png'), raw_right_image)
#     cv2.imwrite(os.path.join(left_save_dir, f'{frame_idx:04d}.png'), raw_left_image)
#     frame_idx +=1

# frame_idx = 0
# while True:
#     try:
#         right_rotated_image, right_pinhole_cw90 = get_undistorted_rotated_image(right_stream_id, frame_idx, right_calib, right_pinhole)
#         left_rotated_image, left_pinhole_cw90 = get_undistorted_rotated_image(left_stream_id, frame_idx, left_calib, left_pinhole)
#         import pdb; pdb.set_trace()
#     except:
#         break
#     cv2.imwrite(f'./coke_slam_right/{frame_idx:04d}.png', right_rotated_image)
#     cv2.imwrite(f'./coke_slam_left/{frame_idx:04d}.png', left_rotated_image)
#     frame_idx +=1


frame_idx = 0
while True:
    try:
        left_rotated_image, left_pinhole_cw90 = get_undistorted_rotated_image(left_stream_id, frame_idx, left_calib, left_pinhole)
        right_rotated_image, right_pinhole_cw90 = get_undistorted_rotated_image(right_stream_id, frame_idx, right_calib, right_pinhole)

        # 
        # Use the rectification transformations to undistort and rectify the images

        # Stereo rectification using OpenCV
        
        # R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K_left, D_left, K_right, D_right, image_size, R_rel, T_rel)
        
        # K_left[:2, 2] += 100 
        # K_right[:2, 2] += np.array([100, 100])
        K_left_copy = K_left.copy()
        K_right_copy = K_right.copy()

        # K_left_copy[:2, 2] += np.array([100, 100])
        # K_right_copy[:2, 2] += np.array([-100, 100])

        # inverse transformation;
        map1x, map1y = cv2.initUndistortRectifyMap(K_left_copy, D_left, R1, P1, image_size, cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(K_right_copy, D_right, R2, P2, image_size, cv2.CV_32FC1)

        # map1x += 100
        # map1y += 100
        # map2x += -100
        # map2y += 100

        # map1x, map1y = cv2.initUndistortRectifyMap(K_left, D_left, R1, P1, image_size, cv2.CV_32FC1)
        # map2x, map2y = cv2.initUndistortRectifyMap(K_right, D_right, R2, P2, image_size, cv2.CV_32FC1)

        rectified_left = cv2.remap(left_rotated_image, map1x, map1y, cv2.INTER_LINEAR)
        rectified_right = cv2.remap(right_rotated_image, map2x, map2y, cv2.INTER_LINEAR)


    except:
        break


    # Test masking
    # Create masks for valid pixels in rectified images
    # mask_left = (map1x >= 0) & (map1x < image_size[1]) & (map1y >= 0) & (map1y < image_size[0])
    # mask_right = (map2x >= 0) & (map2x < image_size[1]) & (map2y >= 0) & (map2y < image_size[0])

    # get mask from the original image  / black area is the invalid area
    mask_left = (rectified_left != 0).astype(np.uint8)
    mask_right = (rectified_right != 0).astype(np.uint8)

    # Find overlapping region by logical AND
    overlap_mask = mask_left & mask_right

    # Dilate each overlap mask to get a rectangular shape
    # kernel = np.ones((10, 10), np.uint8)
    # overlap_dilated = cv2.dilate(overlap_mask.astype(np.uint8), kernel)

    # # Find the bounding rectangle for each dilated mask
    # x, y, w, h = cv2.boundingRect(overlap_dilated)
    
    # new_mask = np.zeros_like(overlap_mask)
    # new_mask[y:y+h, x:x+w] = 1
    new_mask = overlap_mask#overlap_dilated

    # convert the new_mask to 0 and 1, and save it as numpy
    new_mask =  new_mask.astype(np.uint8)
    np.save(os.path.join(save_dir, f'valid_mask.npy'), new_mask)
    # visualize the new_mask
    cv2.imwrite(os.path.join(save_dir, f'valid_mask.png'), new_mask*255)
    break

    # Apply the new masks to the rectified images
    rectified_left_overlap = rectified_left * new_mask.astype(np.uint8)
    rectified_right_overlap = rectified_right * new_mask.astype(np.uint8)


    # Save the overlapping regions (or further process)
    cv2.imwrite(os.path.join(right_save_dir, f'{frame_idx:04d}_overlap.png'), rectified_right_overlap)
    cv2.imwrite(os.path.join(left_save_dir, f'{frame_idx:04d}_overlap.png'), rectified_left_overlap)

    # cv2.imwrite(os.path.join(right_save_dir, f'{frame_idx:04d}.png'), rectified_right)
    # cv2.imwrite(os.path.join(left_save_dir, f'{frame_idx:04d}.png'), rectified_left)
    frame_idx +=1

    if frame_idx > 80:
        break




print(f"Processed total {frame_idx+1} frames ")

