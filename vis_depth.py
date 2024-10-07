
import glob
import numpy as np
import cv2
import torch
import torch.nn as nn

from matplotlib import pyplot as plt
from raftstereo.raft_stereo import *
from threading import Lock
from tqdm import tqdm 


def apply_colormap_and_save(video_frames: np.ndarray, output_path: str, fps: int = 30):
    """
    Apply a colormap to each frame and save the video.
    Args:
        video_frames: 4D numpy array of shape (num_frames, height, width), containing normalized depth maps.
        output_path: Path to save the output video.
        fps: Frames per second for the output video.
    """
    height, width = video_frames.shape[1:3]
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for frame in video_frames:
        # Convert to 8-bit for visualization
        frame_8bit = (frame * 255).astype(np.uint8)
        # Apply a colormap (COLORMAP_JET, COLORMAP_INFERNO, etc.)
        colored_frame = cv2.applyColorMap(frame_8bit, cv2.COLORMAP_JET)
        # Write the frame to the video file
        out.write(colored_frame)

    out.release()

    print(f"Video saved to {output_path}")

def images_to_mp4(image_list, output_file, fps=30):
    """
    Converts a list of images into an mp4 video.

    Args:
        image_list (list of numpy.ndarray): List of images (each image as a numpy array).
        output_file (str): Output file path for the video (e.g., 'output_video.mp4').
        fps (int): Frames per second for the output video.

    Returns:
        None
    """

    # Get the dimensions of the first image
    height, width, layers = image_list[0].shape

    # Define the codec and create a VideoWriter object for mp4 file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 files
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Write each image to the video file
    for i, image in enumerate(image_list):
        if image.shape != (height, width, layers):
            print(f"Image at index {i} has a different size. Skipping it.")
            continue
        out.write(image)
        # cv2.imwrite(f'tmp{i}.png', image)

    # Release the VideoWriter
    out.release()
    print(f"Video saved to {output_file}")


if __name__ == '__main__':
    # Q = np.array(
    #     [[   1.,            0.,            0.,         -251.99191475],
    #     [   0.,            1.,            0.,         -321.33555031],
    #     [   0.,            0.,            0.,          150.        ],
    #     [   0.,            0.,            7.03942946,   -0.        ]]
    # )
    # left - rgb
    Q = np.array(
        [[   1.,            0.,            0.,         -418.3120122 ],
        [   0.,            1.,            0.,         -344.05953693],
        [   0.,            0.,            0.,          150.        ],
        [   0.,            0.,          73.27139128,    0.        ]]
    )

    stereo_depth_model_name = 'eth3d' #'middlebury' # #'middlebury'
    frame_length = 500
    # disparity_min_thr = 4 # below this pixel, likely to be noise from the rectified images' black padding
    # disparity_max_thr = 70 # above this pixel, likely to be noise from the rectified images' black padding
    rgb_depth_fps_ratio = 3

    overlapping_valid_mask = np.load('valid_mask.npy').astype(bool)
    overlapping_valid_mask = np.tile(overlapping_valid_mask[None, :, :], (frame_length, 1, 1))
    # overlapping_valid_mask = np.ones((frame_length, 512, 512), dtype=bool)

    full_flow_est = np.load(f'full_flow_est_{stereo_depth_model_name}.npy')[:frame_length]  
    full_depth_est = np.load(f'full_depth_est_{stereo_depth_model_name}.npy')[:frame_length]  
    # depth_max_thr = Q[2][3] / (Q[3][2] * disparity_min_thr) # above this depth, likely to be noise from the rectified images' black padding
    # depth_min_thr = Q[2][3] / (Q[3][2] * disparity_max_thr) # below this depth, likely to be noise from the rectified images' black padding

    depth_max_thr = 5  # meters
    depth_min_thr = 0.1 # meters
    disparity_min_thr = Q[2][3] / depth_max_thr / Q[3][2]
    disparity_max_thr = Q[2][3] / depth_min_thr / Q[3][2]

    print(f"Depth threshold: {depth_max_thr} meters, {depth_min_thr} meters")
    print(f"Disparity threshold: {disparity_min_thr:.2f} pixels, {disparity_max_thr:.2f} pixels")
    full_flow_est = np.abs(np.array(full_flow_est))
    small_flow_mask = (full_flow_est < disparity_min_thr) 
    valid_mask = (full_flow_est > disparity_min_thr) & (full_flow_est < disparity_max_thr) & overlapping_valid_mask 

    global_min = full_flow_est[valid_mask].min()
    global_max = full_flow_est[valid_mask].max()
    normalized_flow = (full_flow_est - global_min) / (global_max - global_min) * 255
    normalized_flow = normalized_flow.astype(np.uint8)

    normalized_flow[~valid_mask] = 0
    normalized_flow[overlapping_valid_mask & small_flow_mask] = 0

    normalized_flow = np.tile(normalized_flow[..., None], (1, 1, 3))
    # images_to_mp4(normalized_flow, 'flow_video_full.mp4', fps=10)

    # output_path = "colorful_flow_video.mp4"
    # apply_colormap_and_save(normalized_flow, output_path,fps=10)

    full_depth_est = np.abs(np.array(full_depth_est))
    deep_depth_mask = (full_depth_est > depth_max_thr)
    valid_mask = (full_depth_est < depth_max_thr) & (full_depth_est > depth_min_thr) & overlapping_valid_mask 
    global_min = full_depth_est[valid_mask].min()
    global_max = full_depth_est[valid_mask].max()

    normalized_depth = (full_depth_est - global_min) / (global_max - global_min) * 255
    normalized_depth = normalized_depth.astype(np.uint8) 


    normalized_depth[~valid_mask] = 0
    normalized_depth[overlapping_valid_mask & deep_depth_mask] = 255

    # make closer one whiter
    normalized_depth[overlapping_valid_mask] = 255 - normalized_depth[overlapping_valid_mask]

    normalized_depth = np.tile(normalized_depth[..., None], (1, 1, 3))

    # images_to_mp4(normalized_depth, 'depth_video_full.mp4', fps=10)
    print("Finished normalizing depth")

    # # Apply colormap and save the video
    # output_path = "colorful_depth_video.mp4"
    # apply_colormap_and_save(normalized_depth, output_path, fps=10)
    # # Make a video from the depth images   
    
    # read rectified images
    # left_img_path_list = sorted(glob.glob('./data/rectified/coke_slam_left/*.png'))
    # right_img_path_list = sorted(glob.glob('./data/rectified/coke_slam_right/*.png'))
    left_img_path_list = sorted(glob.glob('./data/rectified_left_rgb/coke_slam_left/*.png'))
    right_img_path_list = sorted(glob.glob('./data/rectified_left_rgb/coke_rgb/*.png'))


    # read rgb images from disk
    # rgb is 30 fps, depth is 10 fps, so we need to subsample the rgb video to match the depth video
    # rgb_frame_list = []
    # rgb_video_frames = sorted(glob.glob('./data/rgb/*.png'), key=lambda x: int(x.split('_')[-1].split('.')[0]))[:frame_length*rgb_depth_fps_ratio]
    # for frame_idx, frame_path in enumerate(rgb_video_frames):
    #     # subsample the rgb video to match the depth video
    #     if frame_idx % rgb_depth_fps_ratio == 0:
    #         rgb_frame = cv2.imread(frame_path)
    #         rgb_frame_list.append(rgb_frame)

    print("Finished reading rgb images")
    # Make a video of a grid that shows the rgb, rectified left images, depth, and flow images, while each cell have text label
    grid_video_frames = []
    for i in range(frame_length):
        # rgb_frame = rgb_frame_list[i]
        left_rectified_frame = cv2.imread(left_img_path_list[i])
        rgb_rectified_frame = cv2.imread(right_img_path_list[i])  

        depth_frame = normalized_depth[i]
        flow_frame = normalized_flow[i]
        colored_flow_frame = cv2.applyColorMap(flow_frame, cv2.COLORMAP_JET)

        # Resize the rgb frame to the same size as the depth frame
        # rgb_frame = cv2.resize(rgb_frame, (depth_frame.shape[1], depth_frame.shape[0]))

        # Create a 2x2 grid of images
        grid = np.zeros((2 * depth_frame.shape[0], 2 * depth_frame.shape[1], 3), dtype=np.uint8)

        #  rgb, rectified left images, depth, and flow images
        grid[0:depth_frame.shape[0], 0:depth_frame.shape[1]] = left_rectified_frame #rgb_frame
        grid[0:depth_frame.shape[0], depth_frame.shape[1]:] = rgb_rectified_frame # left_rectified_frame 
        grid[depth_frame.shape[0]:, 0:depth_frame.shape[1]] = depth_frame
        grid[depth_frame.shape[0]:, depth_frame.shape[1]:] = colored_flow_frame

        # Add text labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (255, 255, 255)  # White color
        font_thickness = 1

        grid_height, grid_width, _ = grid.shape
        # Depth label
        left_label = f"Left"
        right_label = f"RGB"
        depth_label = f"Depth"
        flow_label = f"Flow"

        # Put the labels 
        cv2.putText(grid, left_label, (10, 30), font, font_scale, font_color, font_thickness)
        cv2.putText(grid, right_label, (10 + grid_width // 2, 30), font, font_scale, font_color, font_thickness)
        cv2.putText(grid, depth_label, (10, 30 + grid_height // 2), font, font_scale, font_color, font_thickness)
        cv2.putText(grid, flow_label, (10 + grid_width // 2, 30 + grid_height // 2), font, font_scale, font_color, font_thickness)
        grid_video_frames.append(grid)

    images_to_mp4(grid_video_frames, f'grid_video_{stereo_depth_model_name}.mp4', fps=10)
    print("Finished making grid video")
    
    
    
    print('done')

    
