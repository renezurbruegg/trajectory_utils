#   ---------------------------------------------------------------------------------
#   Copyright (c) Microsoft Corporation. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   ---------------------------------------------------------------------------------
"""This is a sample python file for testing functions from the source code."""
from __future__ import annotations

import torch
from trajectory_utils.trajectory import Trajectory


def properties_test():

    positions = torch.rand(10, 3)
    orientations = torch.rand(10, 4)
    orientations = orientations / orientations.norm(dim=1, keepdim=True)
    timesteps = torch.arange(10)
    parent_frame = "parent"
    child_frame = "child"
    traj = Trajectory(positions, orientations, timesteps, parent_frame, child_frame)

    assert traj.positions.equal(positions)
    assert traj.orientations.equal(orientations)
    assert traj.timesteps.equal(timesteps)
    assert traj.start_time == 0
    assert traj.end_time == 9
    assert traj.duration == 9
    assert traj.parent_frame == parent_frame
    assert traj.child_frame == child_frame
    assert traj.euclidean_length == torch.linalg.norm(positions[1:] - positions[:-1], dim=1).sum()


def slicing_test_no_interpolation():
    positions = torch.rand(10, 3)
    orientations = torch.rand(10, 4)
    orientations = orientations / orientations.norm(dim=1, keepdim=True)
    timesteps = torch.arange(10)
    parent_frame = "parent"
    child_frame = "child"
    traj = Trajectory(positions, orientations, timesteps, parent_frame, child_frame)
    # do slicing
    sliced_traj = traj.slice(1.0, 5.0)
    assert sliced_traj.start_time == 1.0
    assert sliced_traj.end_time == 5.0
    sliced_traj = traj.slice(0.0, 7.5)
    assert sliced_traj.start_time == 0.0
    assert sliced_traj.end_time == 7.0


def test_get_item():
    positions = torch.rand(10, 3)
    orientations = torch.rand(10, 4)
    orientations = orientations / orientations.norm(dim=1, keepdim=True)
    timesteps = torch.arange(10)
    parent_frame = "parent"
    child_frame = "child"
    traj = Trajectory(positions, orientations, timesteps, parent_frame, child_frame)
    # do slicing
    sliced_traj = traj[1:5]
    assert sliced_traj.start_time == 1.0
    assert sliced_traj.end_time == 4.0
    sliced_traj = traj[0:7]
    assert sliced_traj.start_time == 0.0
    assert sliced_traj.end_time == 6.0
    sliced_traj = traj[0]
    assert sliced_traj.start_time == 0.0
    assert sliced_traj.end_time == 0.0


def test_resample_nn():
    positions = torch.rand(10, 3)
    orientations = torch.rand(10, 4)
    orientations = orientations / orientations.norm(dim=1, keepdim=True)
    timesteps = torch.arange(10)
    parent_frame = "parent"
    child_frame = "child"
    traj = Trajectory(positions, orientations, timesteps, parent_frame, child_frame)
    # do resampling
    resampled_traj = traj.resample(torch.arange(0, 9.5, 0.5))
    assert resampled_traj.start_time == 0.0
    assert resampled_traj.end_time == 9.0
    assert resampled_traj.timesteps.equal(torch.arange(0, 9.5, 0.5))

    # check if the euclidean length is almost the same
    assert abs(resampled_traj.euclidean_length - traj.euclidean_length) < 1e-6


def test_resample_linear():
    positions = torch.rand(10, 3)
    orientations = torch.rand(10, 4)
    orientations = orientations / orientations.norm(dim=1, keepdim=True)
    timesteps = torch.arange(10)
    parent_frame = "parent"
    child_frame = "child"
    traj = Trajectory(positions, orientations, timesteps, parent_frame, child_frame)
    # do resampling
    resampled_traj = traj.resample(torch.arange(0, 9.5, 0.5), interpolation="linear")
    assert resampled_traj.start_time == 0.0
    assert resampled_traj.end_time == 9.0
    assert resampled_traj.timesteps.equal(torch.arange(0, 9.5, 0.5))
    assert resampled_traj[::2].positions.equal(traj.positions)


def check_show():
    n_pts = 10
    positions = torch.rand(n_pts, 3)
    positions[:, -1] = torch.arange(n_pts)  # make consistant in z direction
    orientations = torch.zeros(n_pts, 4)
    orientations[:, -1] = 1
    orientations = orientations / orientations.norm(dim=1, keepdim=True)
    timesteps = torch.arange(n_pts)
    parent_frame = "parent"
    child_frame = "child"
    traj = Trajectory(positions, orientations, timesteps, parent_frame, child_frame)
    figure = traj.show(show=False)
    resampled_traj = traj.resample(torch.arange(0, n_pts, 0.5), interpolation="linear")
    resampled_traj.show(figure)

def check_transform():
    positions = torch.rand(10, 3)
    positions[:, -1] = torch.arange(10)  # make consistant in z direction
    orientations = torch.rand(10, 4)
    orientations = orientations / orientations.norm(dim=1, keepdim=True)
    timesteps = torch.arange(10)
    parent_frame = "parent"
    child_frame = "child"
    traj = Trajectory(positions, orientations, timesteps, parent_frame, child_frame)

    out = traj.clone().transform(torch.zeros(3) + 0.5, torch.tensor([0, 0, 0, 1]))
    fig = traj.show(show=False)
    out.show(fig)

def check_matmul():
    n_pts = 20
    positions = torch.rand(n_pts, 3)
    positions[:, -1] = torch.arange(n_pts)  # make consistant in z direction
    orientations = torch.rand(n_pts, 4)
    orientations = orientations / orientations.norm(dim=1, keepdim=True)
    timesteps = torch.arange(n_pts)
    parent_frame = "parent"
    child_frame = "child"
    traj = Trajectory(positions, orientations, timesteps, parent_frame, child_frame)
    offset = positions * 0
    offset[:, 1] = 0.05
    orientations = torch.rand(n_pts, 4)
    child_traj = Trajectory(offset, orientations, timesteps, child_frame, "ee").clone()
    fig = traj.show(show=False, line_color="red")
    traj = traj.clone() @ child_traj.clone()
    traj.show(fig)

def check_matmul_different_ts():
    n_pts = 20
    positions = torch.rand(n_pts, 3)
    positions[:, -1] = torch.arange(n_pts)  # make consistant in z direction
    orientations = torch.rand(n_pts, 4)
    orientations = orientations / orientations.norm(dim=1, keepdim=True)
    timesteps = torch.arange(n_pts)
    parent_frame = "parent"
    child_frame = "child"
    traj = Trajectory(positions, orientations, timesteps, parent_frame, child_frame)
    # resample the child trajectory to have different timesteps
    traj = traj.resample(frequency=2, interpolation="linear")
    offset = positions * 0
    offset[:, 1] = 0.05
    orientations = torch.rand(n_pts, 4)
    child_traj = Trajectory(offset, orientations, timesteps, child_frame, "ee").clone()
    fig = traj.show(show=False, line_color="red")
    traj = traj.clone() @ child_traj.clone()
    traj.show(fig)

def check_inverse():
    n_pts = 20
    positions = torch.rand(n_pts, 3)
    positions[:, -1] = torch.arange(n_pts)  # make consistant in z direction
    orientations = torch.rand(n_pts, 4)
    orientations = orientations / orientations.norm(dim=1, keepdim=True)
    timesteps = torch.arange(n_pts)
    parent_frame = "parent"
    child_frame = "child"
    traj = Trajectory(positions, orientations, timesteps, parent_frame, child_frame)
    offset = positions * 0
    offset[:, 1] = 0.05
    orientations = torch.rand(n_pts, 4)
    fig = traj.show(show=False, line_color="red")
    traj = traj.clone() @ traj.inverse().clone()
    traj.show(fig)


def check_temporal_alignment():
    n_pts = 20
    positions = torch.rand(n_pts, 3)
    positions[:, -1] = torch.arange(n_pts)  # make consistant in z direction
    orientations = torch.rand(n_pts, 4)
    orientations = orientations / orientations.norm(dim=1, keepdim=True)
    timesteps = torch.arange(n_pts)
    parent_frame = "parent"
    child_frame = "child"
    orig = Trajectory(positions, orientations, timesteps, parent_frame, child_frame)
    missaligned = Trajectory(positions[1:] + 0.1, orientations[1:], timesteps[1:] + 1, parent_frame, child_frame)
    missaligned = missaligned.slice(missaligned.start_time, missaligned.start_time + 10)

    fig = orig.show(show=False, line_color="red")
    fig = missaligned.show(fig, line_color="blue", show=False)
    # get errors
    relative_trajectory = orig.inverse() @ missaligned
    print("RMSE Position", torch.sqrt(torch.mean((relative_trajectory.positions ** 2).sum(-1))).item())
    # Align without temporal alignment
    aligned_no_temp = missaligned.clone().resample(orig.timesteps).spatial_align(orig)
    relative_trajectory = orig.inverse() @ aligned_no_temp
    print("RMSE Position After aligning", torch.sqrt(torch.mean((relative_trajectory.positions ** 2).sum(-1))).item())
    # temporal alignment
    reference, aligned = orig.clone().temporal_align(missaligned)
    print("RMSE after temporal alignment", torch.sqrt(torch.mean((aligned.positions - reference.positions) ** 2)).item())

    fig = reference.show(fig, show=False, line_color="yellow")
    fig = aligned.show(fig, line_color="green", show=True)



    fig.show()

check_temporal_alignment()
