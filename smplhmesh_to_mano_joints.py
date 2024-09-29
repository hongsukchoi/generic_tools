import os.path as osp
import pickle
import glob
import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import trimesh



def get_fingertips_from_smplhmesh(smplh_mesh: np.ndarray):
    # 6890 x 3
    # order: thumb index middle ring pinky; like here: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker 

    # left: 2748, 2320, 2446, 2557, 2674 [ thumb => pinky ]
    # right: 6214, 5781, 5907, 6016, 6133 [ thumb => pinky ]

    left_fingertips = smplh_mesh[:, [2748, 2320, 2446, 2557, 2674], :]
    right_fingertips = smplh_mesh[:, [6214, 5781, 5907, 6016, 6133], :]

    return left_fingertips, right_fingertips

# mano_left_hand_joints
# 'left_index_1',
# 'left_index_2',
# 'left_index_3',
# 'left_middle_1',
# 'left_middle_2',
# 'left_middle_3',
# 'left_pinky_1',
# 'left_pinky_2',
# 'left_pinky_3',
# 'left_ring_1',
# 'left_ring_2',
# 'left_ring_3',
# 'left_thumb_1',
# 'left_thumb_2',
# 'left_thumb_3',

# mano_right_hand_joints
# 'right_index_1',
# 'right_index_2',
# 'right_index_3',
# 'right_middle_1',
# 'right_middle_2',
# 'right_middle_3',
# 'right_pinky_1',
# 'right_pinky_2',
# 'right_pinky_3',
# 'right_ring_1',
# 'right_ring_2',
# 'right_ring_3',
# 'right_thumb_1',
# 'right_thumb_2',
# 'right_thumb_3',

def reorder_mano_hand_joints(wrist, mano_joints, fingertips):
    """
    Reorder MANO joints to follow the structure of:
    ('Wrist', 'Thumb_1', 'Thumb_2', 'Thumb_3', 'Thumb_4', 'Index_1', 'Index_2', 'Index_3', 'Index_4', 
     'Middle_1', 'Middle_2', 'Middle_3', 'Middle_4', 'Ring_1', 'Ring_2', 'Ring_3', 'Ring_4', 
     'Pinky_1', 'Pinky_2', 'Pinky_3', 'Pinky_4')
    
    Args:
        wrist: Wrist joint (shape: T x 3).
        mano_joints: MANO hand joints (shape: T x 15 x 3).
        fingertips: Fingertip positions (shape: T x 5 x 3).
    
    Returns:
        Reordered hand joints (shape: T x 21 x 3).
    """
    # Reordering indices for MANO joints and finger tips
    # Thumb: index 12, 13, 14 in MANO -> Thumb_1, Thumb_2, Thumb_3
    # Index: index 0, 1, 2 in MANO -> Index_1, Index_2, Index_3
    # Middle: index 3, 4, 5 in MANO -> Middle_1, Middle_2, Middle_3
    # Ring: index 9, 10, 11 in MANO -> Ring_1, Ring_2, Ring_3
    # Pinky: index 6, 7, 8 in MANO -> Pinky_1, Pinky_2, Pinky_3
    reordered_joints = np.concatenate([
        wrist[:, np.newaxis],              # Add wrist joint
        mano_joints[:, [12, 13, 14]],      # Thumb_1, Thumb_2, Thumb_3
        fingertips[:, 0:1],                # Thumb_4 (tip)
        mano_joints[:, [0, 1, 2]],         # Index_1, Index_2, Index_3
        fingertips[:, 1:2],                # Index_4 (tip)
        mano_joints[:, [3, 4, 5]],         # Middle_1, Middle_2, Middle_3
        fingertips[:, 2:3],                # Middle_4 (tip)
        mano_joints[:, [9, 10, 11]],       # Ring_1, Ring_2, Ring_3
        fingertips[:, 3:4],                # Ring_4 (tip)
        mano_joints[:, [6, 7, 8]],         # Pinky_1, Pinky_2, Pinky_3
        fingertips[:, 4:5]                 # Pinky_4 (tip)
    ], axis=1)
    
    return reordered_joints

def plot_3d_joints(joints, title="Hand Joints"):
    """
    Plots the hand joints in 3D.
    
    Args:
        joints (numpy.ndarray): A (21, 3) array of joint positions (x, y, z).
        title (str): Title of the plot.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot joints as scatter points
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c='r', s=50, label='Joints')
    
    # Connect joints with lines (based on the joint hierarchy you defined)
    joint_connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
        (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
    ]
    
    # Draw lines between connected joints
    for start, end in joint_connections:
        ax.plot([joints[start, 0], joints[end, 0]],
                [joints[start, 1], joints[end, 1]],
                [joints[start, 2], joints[end, 2]], 'b')
    
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title(title)
    
    # Equal scale for all axes
    max_range = np.array([joints[:, 0].max() - joints[:, 0].min(), 
                          joints[:, 1].max() - joints[:, 1].min(), 
                          joints[:, 2].max() - joints[:, 2].min()]).max() / 2.0

    mid_x = (joints[:, 0].max() + joints[:, 0].min()) * 0.5
    mid_y = (joints[:, 1].max() + joints[:, 1].min()) * 0.5
    mid_z = (joints[:, 2].max() + joints[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()

if __name__ == '__main__':

    to_save_path = 'hand_joints_brent.pkl'
    data_path = './hongsuk_data.npz'
    data = np.load(data_path, allow_pickle=True)
    
    
    smplh_mesh_verts = data["verts"] # T x 6890 x 3
    smplh_joints = data["joint_positions"] # T x 52 x 3
    
    left_wrist, right_wrist = smplh_joints[:, 20], smplh_joints[:, 21]
    mano_left_hand_joints = smplh_joints[:, 22:37]
    mano_right_hand_joints = smplh_joints[:, 37:52]
    left_fingertips, right_fingertips = get_fingertips_from_smplhmesh(smplh_mesh_verts)

    # Reorder the left hand joints
    full_mano_left_hand_joints = reorder_mano_hand_joints(left_wrist, mano_left_hand_joints, left_fingertips)
    
    # Reorder the right hand joints
    full_mano_right_hand_joints = reorder_mano_hand_joints(right_wrist, mano_right_hand_joints, right_fingertips)
    # plot_3d_joints(full_mano_left_hand_joints[0])
    # plot_3d_joints(full_mano_right_hand_joints[110])
    
    hand_joints = {}
    hand_joints['right_hand'] = full_mano_right_hand_joints
    hand_joints['left_hand'] = full_mano_left_hand_joints

    with open(f'{to_save_path}', 'wb') as f:
        pickle.dump(hand_joints, f)
