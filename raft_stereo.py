
import glob
import numpy as np
import cv2
import torch
import torch.nn as nn

from matplotlib import pyplot as plt
from raftstereo.raft_stereo import *
from threading import Lock
from tqdm import tqdm



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



class StereoDepth:

    def __init__(self, width: int, height: int, model_name: str):
        self.width = 512
        self.height = 512
        # left - right
        # self.Q = np.array(
        #     [[   1.,            0.,            0.,         -251.99191475],
        #     [   0.,            1.,            0.,         -321.33555031],
        #     [   0.,            0.,            0.,          150.        ],
        #     [   0.,            0.,            7.03942946,   -0.        ]]
        # ) 

        # left - rgb
        self.Q = np.array(
            [[   1.,            0.,            0.,         -418.3120122 ],
            [   0.,            1.,            0.,         -344.05953693],
            [   0.,            0.,            0.,          150.        ],
            [   0.,            0.,          73.27139128,    0.        ]]
        )

        self.left_img_path_list = sorted(glob.glob('./data/rectified_left_rgb/coke_slam_left/*.png'))
        self.right_img_path_list = sorted(glob.glob('./data/rectified_left_rgb/coke_rgb/*.png'))

        # Create lock for raft -- gpu threading messes up CUDA memory state, with curobo...
        self.raft_lock = Lock()
        with self.raft_lock:
            self.model = create_raft(model_name=model_name)
        

    def get_depth(
        self, idx
    ):
     
        # Step 1: Read the grayscale image
        # Step 2: Convert the grayscale image to an RGB image
        left_gray_image = cv2.imread(self.left_img_path_list[idx], cv2.IMREAD_GRAYSCALE)
        left_rgb = cv2.cvtColor(left_gray_image, cv2.COLOR_GRAY2RGB)

        right_gray_image = cv2.imread(self.right_img_path_list[idx], cv2.IMREAD_GRAYSCALE)
        right_rgb = cv2.cvtColor(right_gray_image, cv2.COLOR_GRAY2RGB)

        # right_gray_image = cv2.imread(self.right_img_path_list[idx])
        # right_gray_image = right_gray_image.mean(axis=2)
        # right_rgb = np.tile(right_gray_image[...,None], (1,1,3))

        # if idx == 5:
        #     cv2.imwrite(f'left_rgb_{idx}.png', left_rgb)
        #     cv2.imwrite(f'right_rgb_{idx}.png', right_rgb * 1.1)
        #     import pdb; pdb.set_trace()

        left,right = torch.from_numpy(np.flip(left_rgb[...,:3],axis=2).copy()).float().cuda(), torch.from_numpy(np.flip(right_rgb[...,:3],axis=2).copy()).float().cuda()
        
        left_torch,right_torch = left.permute(2,0,1),right.permute(2,0,1)
        with self.raft_lock:
            flow = raft_inference(left_torch, right_torch, self.model, 32)
        
        flow = flow.cpu().numpy()
        flow = np.abs(flow)

        points_3D = cv2.reprojectImageTo3D(flow, self.Q) 
        # Extract the Z (depth) component of the 3D points
        depth_map = points_3D[:, :, 2]
        # the same except the current setting has a vertical shift
        # fx = P1[0,0]
        # depth = fx* baseline/(flow.abs()+self.cx_diff)
        # depth = depth.cpu().numpy()

        return depth_map, points_3D, flow
    
if __name__ == '__main__':
    model_name = 'eth3d' #'middlebury' 
    depth_model = StereoDepth(512, 512, model_name)

    full_depth_est = []
    full_points_3D_est = []
    full_flow_est = []
    print("Length: ", len(depth_model.right_img_path_list))
    for i in tqdm(range(len(depth_model.right_img_path_list))):
        # if not (0 <= i <= 10):
        #     continue
        depth_est, points_3D_est, flow = depth_model.get_depth(i)
        full_depth_est.append(depth_est)
        full_flow_est.append(flow)
        full_points_3D_est.append(points_3D_est)

    np.save(f'full_depth_est_{model_name}.npy', full_depth_est)
    np.save(f'full_flow_est_{model_name}.npy', full_flow_est)
    np.save(f'full_points_3D_est_{model_name}.npy', full_points_3D_est)

    # disparity_thr = 2 # below this pixel, likely to be noise from the rectified images' black padding
    # depth_thr = depth_model.Q[2][3] / depth_model.Q[3][2] / disparity_thr # above this depth, likely to be noise from the rectified images' black padding

    # print(f"Disparity threshold: {disparity_thr}")
    # print(f"Depth threshold: {depth_thr}")
    # full_flow_est = np.abs(np.array(full_flow_est))
    # valid_mask = full_flow_est > disparity_thr
    # global_min = full_flow_est[valid_mask].min()
    # global_max = full_flow_est[valid_mask].max()
    # normalized_flow = (full_flow_est - global_min) / (global_max - global_min) * 255
    # normalized_flow = normalized_flow.astype(np.uint8)
    # normalized_flow[~valid_mask] = 0
    # normalized_flow = np.tile(normalized_flow[..., None], (1, 1, 3))

    # images_to_mp4(normalized_flow, 'flow_video_full.mp4', fps=10)

    # full_depth_est = np.abs(np.array(full_depth_est))
    # valid_mask = np.ones_like(full_depth_est).astype(bool) #full_depth_est < depth_thr
    # global_min = full_depth_est[valid_mask].min()
    # global_max = full_depth_est[valid_mask].max()

    # normalized_depth = (full_depth_est - global_min) / (global_max - global_min) * 255
    # normalized_depth = normalized_depth.astype(np.uint8) 
    # # normalized_depth[~valid_mask] = 0
    # normalized_depth = np.tile(normalized_depth[..., None], (1, 1, 3))
    # images_to_mp4(normalized_depth, 'depth_video_full.mp4', fps=10)


    # Make a video from the depth images   
    print('done')

    
