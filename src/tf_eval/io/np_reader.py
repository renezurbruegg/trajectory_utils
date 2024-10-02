from __future__ import annotations


import rosbag
import os
import pandas as pd
from tf_eval.trajectory import Trajectory

from prettytable import PrettyTable
import numpy as np
import torch
import roma

def trajectory_from_numpy(trajectory_path:str, timestamp_path:str, child_frame:str, parent_frame:str, timestamp_unit = 1e9) -> Trajectory:
    poses = torch.from_numpy(np.load(trajectory_path)).float()
    orientations = torch.zeros(len(poses), 4).to(poses.device)
    orientations[:, -1] = 1
    positions = torch.zeros(len(poses), 3).to(poses.device)

    if len(poses.shape) == 3:
        # this are rotation matrices
        orientations = roma.rotmat_to_unitquat(poses[..., :3, :3])
        positions = poses[..., :3, -1]
    elif len(poses.shape) == 2:
        if poses.shape[-1] == 3:
            positions = poses
            print(f"[WARNING] No orientation provided, using identity quaternion for {parent_frame}->{child_frame}")
        elif poses.shape[-1] == 7:
            positions = poses[..., :3]
            orientations = poses[..., 3:]
        else:
            raise ValueError(f"Invalid shape for poses: {poses.shape}")

    stamps = np.load(timestamp_path)
    stamps = torch.from_numpy(stamps).to(dtype=torch.float64) / timestamp_unit # ns -> seconds
    return Trajectory(positions, orientations, stamps, parent_frame, child_frame)
