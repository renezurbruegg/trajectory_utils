# pyright: reportRedeclaration=false
from __future__ import annotations

import torch

import roma
from tf_eval.trajectory import Trajectory
from prettytable import PrettyTable
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def compare_trajectories(traj1: Trajectory, traj2: Trajectory, headless: bool = False) -> dict[str, float]:
    
    if traj1.child_frame != traj2.child_frame:
        raise ValueError("Trajectories have different child frames, got {} and {}".format(traj1.child_frame, traj2.child_frame))
    if traj1.parent_frame != traj2.parent_frame:
        raise ValueError("Trajectories have different parent frames, got {} and {}".format(traj1.parent_frame, traj2.parent_frame))

    # align the two trajectories
    # traj1 = traj1.spatial_align(traj2, orientation=False)

    # subtract the two trajectories
    relative_trajectory = traj1.inverse() @ traj2

    relative_rotation_vector = roma.unitquat_to_rotvec(relative_trajectory.orientations).norm(dim=1)
    rmse_rotation = torch.sqrt(torch.mean((relative_rotation_vector ** 2))).item()


    rmse_position = torch.sqrt(torch.mean((relative_trajectory.positions ** 2).sum(dim=1))).item()
    data = {
        "parent_frame": traj1.parent_frame,
        "child_frame": traj1.child_frame,
        "duration": relative_trajectory.duration,
        "total_translation_error": relative_trajectory.euclidean_length,
        "rmse_position": rmse_position,
        "rmse_rotation": rmse_rotation,
    }

    # Calculate accuracy for different translation and rotation thresholds
    for t, r in zip([1e-2, 3e-2, 5e-2, 10e-2], [1, 3, 5, 10]):
        rotation_error_deg = relative_rotation_vector * 180 / torch.pi
        position_error_m = relative_trajectory.positions.norm(dim=1)
        acc = (position_error_m < t) & (rotation_error_deg < r)
        acc = acc.float().mean().item()
        data[f"acc_{100*t:.0f}cm_{r}°"] = acc

    if not headless:
        # create plot, with 3 subplots using plotly
        fig = make_subplots(rows=3, cols=1, subplot_titles=("Translation", "Translation", "Rotation"), specs =
            [[{"type": "scatter3d"}], [{"type": "xy"}], [{"type": "xy"}]],
        )

        # set Title
        fig.update_layout(title=f"Trajectory Comparison from {traj1.parent_frame} to {traj1.child_frame}")

        fig  = traj1.show(fig=fig, show=False, line_color="blue", show_frames=True, trace_kwargs={"row": 1, "col": 1}, time_as_color = True, colorscale="Viridis")
        fig = traj2.show(fig=fig, show=False, line_color="red", show_frames=True, trace_kwargs={"row": 1, "col": 1}, time_as_color = True, colorscale="Viridis")

        fig.add_trace(
            go.Scatter(x=relative_trajectory.timesteps, y=relative_trajectory.positions.norm(dim=1), mode="lines", name="Translation Error"),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(x=relative_trajectory.timesteps, y=relative_rotation_vector, mode="lines", name="Rotation Error"),
            row=3, col=1
        )
        fig.show()

    table = PrettyTable()
    table.field_names = ["Property", "Value"]
    table.align = "l"
    for key, value in data.items():
        table.add_row([key, value])
    print(table)
    return data
