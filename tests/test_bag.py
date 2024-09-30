#   ---------------------------------------------------------------------------------
#   Copyright (c) Microsoft Corporation. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   ---------------------------------------------------------------------------------
"""This is a sample python file for testing functions from the source code."""
from __future__ import annotations

import torch
from tf_eval.bag_converter import convert_rosbag_to_dfs
from tf_eval.eval import compare_trajectories
from tf_eval.trajectory import Trajectory

dfs = convert_rosbag_to_dfs("data/testing_gt_2024-09-27-18-19-41.bag", output_dir="data", save=False)
trajectories = {}
for child_frame, df in dfs.items():
    positions = torch.from_numpy(df[["translation_x", "translation_y", "translation_z"]].values).float()
    orientations = torch.from_numpy(df[["rotation_x", "rotation_y", "rotation_z", "rotation_w"]].values).float()
    timesteps = torch.from_numpy(df["timestamp"].values).double()
    parent_frame = df["parent_frame"].values[0]
    child_frame = df["child_frame"].values[0]
    traj = Trajectory(positions, orientations, timesteps, parent_frame, child_frame).resample(frequency=10)
    trajectories[child_frame] =  traj

colormap = {
    "vicon/RigidBody/RigidBody": "blue",
    "vicon/AriaGlasses/AriaGlasses": "black",
    "vicon/MarkerOrigin/MarkerOrigin": "green",
}

# Convert everything in marker origin frame
base_traj = trajectories["vicon/MarkerOrigin/MarkerOrigin"].clone().average().inverse().resample(frequency=10, start_time=trajectories["vicon/MarkerOrigin/MarkerOrigin"].start_time, end_time=trajectories["vicon/MarkerOrigin/MarkerOrigin"].end_time)
for key in trajectories.keys():
    trajectories[key] = base_traj @ trajectories[key]

# Start evaluation
# gt trajectories in world frame
gt_w_object_traj = trajectories["vicon/RigidBody/RigidBody"]
gt_w_glasses_traj = trajectories["vicon/AriaGlasses/AriaGlasses"]

# gt trajectories in glasses frame
gt_g_object_traj = trajectories["vicon/AriaGlasses/AriaGlasses"].inverse() @ trajectories["vicon/RigidBody/RigidBody"]

# TODO, get actual prediction values
pred_w_object_traj = trajectories["vicon/RigidBody/RigidBody"].clone()
pred_w_object_traj._positions += torch.rand_like(pred_w_object_traj._positions) * 0.01
pred_w_object_traj._orientations += torch.rand_like(pred_w_object_traj._orientations) * 0.01
pred_w_object_traj._orientations /= pred_w_object_traj._orientations.norm(dim=1, keepdim=True)

# TODO, get actual prediction values
pred_w_glases_traj = trajectories["vicon/AriaGlasses/AriaGlasses"].clone()
pred_w_glases_traj._positions += torch.rand_like(pred_w_glases_traj._positions) * 0.01
pred_w_glases_traj._orientations += torch.rand_like(pred_w_glases_traj._orientations) * 0.01
pred_w_glases_traj._orientations /= pred_w_glases_traj._orientations.norm(dim=1, keepdim=True)

# TODO, get actual prediction values
pred_g_object_traj = pred_w_glases_traj.inverse() @ trajectories["vicon/RigidBody/RigidBody"]
pred_g_object_traj._positions += torch.rand_like(pred_g_object_traj._positions) * 0.01
pred_g_object_traj._orientations += torch.rand_like(pred_g_object_traj._orientations) * 0.01
pred_g_object_traj._orientations /= pred_g_object_traj._orientations.norm(dim=1, keepdim=True)

# Evaluate object in glasses frame
print("===== Object in Glasses Frame =====")
compare_trajectories(gt_w_object_traj, pred_w_object_traj)
print("")

# Evaluate glasses in world frame
print("===== Glasses in World Frame =====")
compare_trajectories(gt_w_glasses_traj, pred_w_glases_traj)
print("")

# Evaluate object in glasses frame
print("===== Object in Glasses Frame =====")
compare_trajectories(gt_g_object_traj, pred_g_object_traj)
print("")
