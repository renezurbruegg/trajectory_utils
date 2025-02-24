"""Example of how to load a rosbag and compare the trajectories in it."""

from __future__ import annotations

import torch
from trajectory_utils.io.bag_converter import load_trajectories_from_bag
from trajectory_utils.eval import compare_trajectories
from trajectory_utils.trajectory import Trajectory
from trajectory_utils.io.np_reader import trajectory_from_numpy
from trajectory_utils.utils.colors import get_colormap
import os

import numpy as np
import json

import argparse
argparser = argparse.ArgumentParser(description="Evaluate a prediction against a ground truth trajectory.")
argparser.add_argument("--bag", type=str, default="./data/gt_testing_1.bag", help="Path to the rosbag file.")
argparser.add_argument("--obj_traj", type=str, default="./data/gt_testing_pred/right_hand_positions.npy", help="Path to the object trajectory.")
argparser.add_argument("--obj_ts", type=str, default="./data/gt_testing_pred/right_hand_timestamps.npy", help="Path to the object timestamps.")
argparser.add_argument("--glasses_traj", type=str, default="./data/gt_testing_pred/trajectory.npy", help="Path to the glasses trajectory.")
argparser.add_argument("--glasses_ts", type=str, default="./data/gt_testing_pred/timestamps.npy", help="Path to the glasses timestamps.")

argparser.add_argument("--sampling_frequency", type=int, default=30, help="Sampling frequency for the trajectories.")
argparser.add_argument("--time_sync_slack", type=float, default=0.1, help="Time sync slack for interaction detection. This is the time in seconds that is used for evaluation before and after the interaction.")
argparser.add_argument("--output_folder", type=str, default="data", help="Output path for the evaluation results.")
argparser.add_argument("--output_name", type=str, default=None, help="Output name for the evaluation results. The actual output will be saved as output_folder/output_name.json")
argparser.add_argument("--headless", action="store_true", help="Run the script without showing any plots.")

args = argparser.parse_args()
if args.output_name is None:
    args.output_name = args.bag.split("/")[-1].replace(".bag", "")

################################################
# Helper Functions
################################################

# Smaller helper function to load Prediction trajectories
def get_prediction_trajectories(
        path_to_object_trajectory: str, path_to_object_timestamps: str,
        path_to_glasses_trajectory: str, path_to_glasses_timestamps: str) -> dict[str, Trajectory]:

    """Load prediction trajectories from a numpy file.
    
    Args:
        path_to_prediction (str): Path to the numpy file.
        
    Returns:
        dict: A dictionary of trajectories.
    """
    pred_hand_traj = trajectory_from_numpy(path_to_object_trajectory, path_to_object_timestamps, child_frame="Prediction/RigidBody", parent_frame="world")
    pred_glasses_traj = trajectory_from_numpy(path_to_glasses_trajectory, path_to_glasses_timestamps, child_frame="RigidBody/AriaGlasses", parent_frame="world")
    return {"Prediction/RigidBody": pred_hand_traj, "RigidBody/AriaGlasses": pred_glasses_traj}

def trim_trajectories_to_interaction(pred_obj: Trajectory, pred_glasses: Trajectory, gt_obj: Trajectory, gt_glasses: Trajectory) -> tuple[Trajectory, Trajectory, Trajectory, Trajectory]:
    """Trim object trajectories based on object interaction.
    
    This function assumes the given trajectories to be time-synchronized.
    It then determines start and end of an interaction based on the object trajectory. If the object is moving more than 0.05 m/s, it is considered an interaction.
    Once the object does not move more than 0.05 m/s, the interaction is considered to be over.

    Args:
        pred_obj (Trajectory): Prediction object trajectory.
        pred_glasses (Trajectory): Prediction glasses trajectory.
        gt_obj (Trajectory): Ground truth object trajectory.
        gt_glasses (Trajectory): Ground truth glasses trajectory.
    
    Returns:
        tuple[Trajectory, Trajectory, Trajectory, Trajectory]: Trimmed trajectories, based on the interaction of the gt_obj trajectory.
    """

    # We use the object trajectory to find the start and end of the interaction
    positions = gt_obj.positions
    positions_1d = positions.norm(dim=1)
    positions_1d_diff = torch.cat([torch.zeros(1, device=positions_1d.device, dtype=positions_1d.dtype), positions_1d.diff() / gt_obj._timesteps.diff().abs() ])
    # slightly smoothen, lowpass filter
    positions_1d_diff = torch.from_numpy(np.convolve(positions_1d_diff.numpy(), np.ones(5)/5, mode="same"))
    # determine interaction as when the object is moving more than 0.05 m/s
    interaction = positions_1d_diff > 0.05
    start_idx = torch.where(interaction)[0][0].item()
    end_idx = torch.where(interaction)[0][-1].item()
    # find start and end of interaction
    start_ts = gt_obj._timesteps[start_idx].item() - args.time_sync_slack
    end_ts = gt_obj._timesteps[end_idx].item() + args.time_sync_slack
    # make sure end_ts is valid match with frequency
    end_ts = end_ts - (end_ts - start_ts) % (1/args.sampling_frequency)

    # Debug information, to show thresholds and interaction start and end
    # plt.plot(gt_object_w._timesteps, 0*gt_object_w._timesteps + 0, label="Object")
    # plt.plot(gt_glasses_w_aligned._timesteps, 0*gt_glasses_w_aligned._timesteps + 1, label="Glasses")
    # plt.plot(pred_object_w._timesteps, 0*pred_object_w._timesteps + 2, label="Pred Object")
    # plt.plot(pred_glasses_w._timesteps, 0*pred_glasses_w._timesteps + 3, label="Pred Glasses")
    # plt.legend()
    # plt.plot([start_ts, start_ts], [-1, 4], label="Interaction Start", color="black")
    # plt.plot([end_ts, end_ts], [-1, 4], label="Interaction End", color="black")
    # plt.show()

    # Finally, slice all trajectories to the interaction
    pred_obj = pred_obj.slice(start_time=start_ts, end_time=end_ts).resample(frequency=args.sampling_frequency, start_time=start_ts, end_time=end_ts).clone()
    pred_glasses = pred_glasses.slice(start_time=start_ts, end_time=end_ts).resample(frequency=args.sampling_frequency, start_time=start_ts, end_time=end_ts).clone()
    gt_obj = gt_obj.slice(start_time=start_ts, end_time=end_ts).resample(frequency=args.sampling_frequency, start_time=start_ts, end_time=end_ts).clone()
    gt_glasses = gt_glasses.slice(start_time=start_ts, end_time=end_ts).resample(frequency=args.sampling_frequency, start_time=start_ts, end_time=end_ts).clone()
    return pred_obj, pred_glasses, gt_obj, gt_glasses



################################################
# Main Script
################################################

# Load data
gt_trajectories = load_trajectories_from_bag(args.bag)
pred_trajectories = get_prediction_trajectories(
    args.obj_traj, args.obj_ts,
    args.glasses_traj, args.glasses_ts
)

# Resample the prediction trajectories at the same frequency
for key, traj in pred_trajectories.items():
    pred_trajectories[key] = traj.resample(frequency=args.sampling_frequency)
for key, traj in gt_trajectories.items():
    gt_trajectories[key] = traj.resample(frequency=args.sampling_frequency)

gt_glasses_w, gt_object_w = gt_trajectories["vicon/AriaGlasses"], gt_trajectories["vicon/RigidBody"]
pred_glasses_w, pred_object_w = pred_trajectories["RigidBody/AriaGlasses"], pred_trajectories["Prediction/RigidBody"]

# align the head trajectories to find time delay and rotation
gt_glasses_w, pred_glasses_w_aligned, infos = gt_glasses_w.clone().temporal_align(pred_glasses_w, return_infos=True)
delay, rotation, translation = infos["delay"], infos["rotation"], infos["translation"]
gt_object_w = gt_object_w.slice(start_time=gt_glasses_w.start_time, end_time=gt_glasses_w.end_time)

# Shift all predictions to achieve spatial and temporal alignment
pred_glasses_w = pred_glasses_w.transform(translation, rotation)
pred_object_w = pred_object_w.transform(translation, rotation)
pred_glasses_w.parent_frame = "vicon"
pred_object_w.parent_frame = "vicon"
pred_glasses_w._timesteps -= delay
pred_object_w._timesteps -= delay

# Trim the trajectories to match the object interaction
pred_object_w, pred_glasses_w, gt_object_w, gt_glasses_w = trim_trajectories_to_interaction(pred_object_w, pred_glasses_w, gt_object_w, gt_glasses_w)

if not args.headless:
    # Show aligned trajectories
    fig = pred_object_w.show(show=False, line_color="blue", time_as_color=get_colormap("B71C1C"))
    fig = gt_object_w.show(fig, show=False, line_color="darkblue", time_as_color=get_colormap("FF1744"))
    fig = pred_glasses_w.show(fig, show=False, line_color="#00C853", time_as_color=get_colormap("00CCFF"))
    fig = gt_glasses_w.show(fig, show=False, line_color="#00CCFF", time_as_color=get_colormap("00BBCC"))
    # Title
    fig.update_layout(title="Aligned Trajectories")
    fig.show()

data = {}
# Eval needed quantities
print("===== Object in World Frame =====")
# Overwrite frame to make sure the comparison does not complain about different frames
pred_object_w.child_frame = "vicon/RigidBody"
data["object_metrics"] = compare_trajectories(gt_object_w, pred_object_w, headless=args.headless)
print("")
print("===== Glasses in World Frame =====")
pred_glasses_w.child_frame = "vicon/AriaGlasses"
data["glasses_metric"] = compare_trajectories(gt_glasses_w, pred_glasses_w, headless=args.headless)
print("")
print("===== Object in Glasses Frame =====")
data["object_in_glasses_metric"] = compare_trajectories(gt_glasses_w.inverse() @ gt_object_w, pred_glasses_w.inverse() @ pred_object_w, headless=args.headless)

# save the data
os.makedirs(args.output_folder, exist_ok=True)
with open(f"{args.output_folder}/{args.output_name}.json", "w") as f:
    json.dump(data, f, indent=4)
    print(f"Saved evaluation results to {args.output_folder}/{args.output_name}.json")