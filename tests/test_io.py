#   ---------------------------------------------------------------------------------
#   Copyright (c) Microsoft Corporation. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   ---------------------------------------------------------------------------------
"""This is a sample python file for testing functions from the source code."""
from __future__ import annotations

import torch
from trajectory_utils.trajectory import Trajectory
import numpy as np
import roma
from trajectory_utils.np_reader import trajectory_from_numpy





# parent_frame = ""
# child_frame = ""
# print("Read numpy")
# poses = np.load("/home/zrene/git/trajectory_utils/data/gt_testing_pred/right_hand_positions.npy")
# stamps = np.load("/home/zrene/git/trajectory_utils/data/gt_testing_pred/right_hand_timestamps.npy")

# poses = torch.from_numpy(poses).float()
# orientations = torch.zeros(len(poses), 4).to(poses.device)
# orientations[:, -1] = 1
# stamps = torch.from_numpy(stamps).to(dtype=torch.float64) / 1000 # ms -> seconds

# fig = Trajectory(poses, orientations, stamps, parent_frame, child_frame).show(show=False)

# poses2 = np.load("/home/zrene/git/trajectory_utils/data/gt_testing_pred/trajectory.npy")
# poses2 = torch.from_numpy(poses2).float()
# orientations = roma.rotmat_to_unitquat(poses2[..., :3, :3])
# positions = poses2[:, :3, -1]
# stamps2 = np.load("/home/zrene/git/trajectory_utils/data/gt_testing_pred/timestamps.npy")
# stamps2 = torch.from_numpy(stamps2).to(dtype=torch.float64) / 1000 
# # 
# fig = Trajectory(positions, orientations, stamps2, parent_frame, child_frame+"_2").show(fig, show=False, line_color = "red")




def check_read_numpy():
    hand_traj = trajectory_from_numpy("/home/zrene/git/trajectory_utils/data/gt_testing_pred/right_hand_positions.npy", "/home/zrene/git/trajectory_utils/data/gt_testing_pred/right_hand_timestamps.npy", child_frame="right_hand", parent_frame="world")
    glasses_traj = trajectory_from_numpy("/home/zrene/git/trajectory_utils/data/gt_testing_pred/trajectory.npy", 
    "/home/zrene/git/trajectory_utils/data/gt_testing_pred/timestamps.npy", child_frame="glasses", parent_frame="world")
    
    fig = hand_traj.show(show=False)
    fig = glasses_traj.show(fig, show=False, line_color = "red")
    fig.show()


check_read_numpy()


# positions = torch.rand(10, 3)
# orientations = torch.rand(10, 4)
# orientations = orientations / orientations.norm(dim=1, keepdim=True)
# timesteps = torch.arange(10)
# parent_frame = "parent"
# child_frame = "child"
# traj = Trajectory(positions, orientations, timesteps, parent_frame, child_frame)
# # do slicing
# sliced_traj = traj[1:5]
# assert sliced_traj.start_time == 1.0
# assert sliced_traj.end_time == 4.0
# sliced_traj = traj[0:7]