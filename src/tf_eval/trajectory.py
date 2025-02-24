# pyright: reportRedeclaration=false
from __future__ import annotations

import torch
from typing import Callable, Sequence, TypeVar
import roma
from trajectory_utils.utils.rotation import unitquat_slerp
import plotly.graph_objects as go


Self = TypeVar('Self', bound='Trajectory')

class Trajectory:
    """Generic class to represent a trajectory in 3D space.

    Methods:
        slice(start_time: float | None, end_time: float | None, interpolate: bool = False) -> Trajectory:
            Slices the trajectory based on the given start and end times.
        inverse() -> Trajectory:
            Returns the inverse of the trajectory.
        transform(position: torch.Tensor, orientation: torch.Tensor) -> Trajectory:
            Transforms the trajectory using the given position and orientation.
        as_rigid_unit_quat() -> roma.RigidUnitQuat:
            Converts the trajectory to a rigid unit quaternion representation.
        clone() -> Trajectory:
            Creates a deep copy of the trajectory.
        resample(new_timesteps: Sequence[float] | torch.Tensor | float | None = None, start_time: float | None = None, end_time: float | None = None, frequency: float | None = None, interpolation: str = "linear") -> Trajectory:
            Resamples the trajectory based on the given parameters.
        show(fig: go.Figure | None = None, show: bool = True, line_color: str = "blue", show_frames = False, frame_scale = 0.05, trace_kwargs={}, time_as_color:bool = False, colorscale = "viridis") -> go.Figure:
            Visualizes the trajectory using a 3D plot.
        temporal_align(other: Trajectory, translation = True, orientation: bool = True, max_delay: float = torch.inf, return_infos = False) -> Trajectory:
            Temporally aligns the trajectory with another trajectory.

    Internal Attributes:
        _positions: torch.Tensor
            The positions of the trajectory.
        _orientations: torch.Tensor
            The orientations of the trajectory.
        _timesteps: torch.Tensor
            The timestamps of the trajectory.
        _parent_frame: str
            The name of the parent frame.
        _child_frame: str
            The name of the child frame.
    
    Properties:
        avg_position: torch.Tensor
            The average position of the trajectory.
        avg_orientation: torch.Tensor
            The average orientation of the trajectory.
        positions: torch.Tensor
            The positions of the trajectory.
        orientations: torch.Tensor
            The orientations of the trajectory.
        timesteps: torch.Tensor
            The timestamps of the trajectory.
        start_time: float
            The start time of the trajectory.
        end_time: float
            The end time of the trajectory.
        duration: float
            The duration of the trajectory.
        parent_frame: str
            The name of the parent frame.
        child_frame: str
            The name of the child frame.
        euclidean_length: float
            The euclidean length of the trajectory.
    
    Args:
        positions (list[torch.Tensor] | torch.Tensor): A list of 3D positions or a tensor of shape (N, 3) representing the positions of the trajectory
        orientations (list[torch.Tensor] | torch.Tensor | None): A list of unit quaternions or a tensor of shape (N, 4) representing the orientations of the trajectory. If None, the orientation is assumed to be the identity quaternion
        timesteps (list[float] | torch.Tensor): A list of timestamps or a tensor of shape (N,) representing the timestamps of the trajectory
        parent_frame (str): The name of the parent frame
        child_frame (str): The name of the child frame
        """

    def __init__(
        self,
        positions: list[torch.Tensor] | torch.Tensor,
        orientations: list[torch.Tensor] | torch.Tensor | None,
        timesteps: list[float] | torch.Tensor,
        parent_frame: str,
        child_frame: str,
    ):

        if not isinstance(positions, torch.Tensor):
            positions = torch.stack(positions)
        self._positions: torch.Tensor = positions

        if not isinstance(orientations, torch.Tensor):
            if orientations is None:
                orientations: torch.Tensor = torch.zeros(len(self._positions), 4)
                orientations[:, 3] = 1
            else:
                orientations = torch.stack(orientations)
        self._orientations: torch.Tensor = orientations

        if len(self._positions) != len(self._orientations):
            raise ValueError(
                "The number of positions and orientations must be equal. Got {} positions and {} orientations".format(
                    len(self._positions), len(self._orientations)
                )
            )

        if not isinstance(timesteps, torch.Tensor):
            timesteps = torch.tensor(timesteps)

        if timesteps.dtype != torch.float64:
            timesteps = timesteps.to(torch.float64)

        if len(timesteps) != len(self._positions):
            raise ValueError(
                "The number of timesteps must be equal to the number of positions. Got {} timesteps and {} positions".format(
                    len(timesteps), len(self._positions)
                )
            )

        self._timesteps = timesteps
        self._parent_frame = parent_frame
        self._child_frame = child_frame


    def slice(self, start_time: float | None, end_time: float | None, interpolate: bool = False) -> Trajectory:
        """
        Slice the trajectory based on the given start and end times.
        Args:
            start_time (float | None): The start time of the slice. If None, the start time of the trajectory will be used.
            end_time (float | None): The end time of the slice. If None, the end time of the trajectory will be used.
            interpolate (bool, optional): Whether to interpolate the sliced trajectory. Defaults to False.
        Returns:
            Trajectory: The sliced trajectory.
        Raises:
            NotImplementedError: If interpolation is requested but not yet implemented.
        """

        if interpolate:
            raise NotImplementedError("Interpolation is not yet implemented")
        if start_time is None:
            start_time = self.start_time
        if end_time is None:
            end_time = self.end_time

        start_idx = torch.searchsorted(self._timesteps, start_time, right=False)
        end_idx = torch.searchsorted(self._timesteps, end_time, right=True)
        return Trajectory(
            self._positions[start_idx:end_idx],
            self._orientations[start_idx:end_idx],
            self._timesteps[start_idx:end_idx],
            self._parent_frame,
            self._child_frame,
        )

    def __getitem__(self, key) -> Trajectory:
        """Slice the trajectory based on the given key.

        Note, this will spaces strictly based on the index and not the time.

        Example:
            traj = Trajectory(positions, orientations, timesteps, parent_frame, child_frame)
            sliced_traj = traj[1:5]
            # -> This will slice the trajectory from index 1 to 5

        Args:
            key: The key to slice the trajectory. Either an integer or a slice object.
        
        Returns:
            Trajectory: The sliced trajectory.

        """
        if isinstance(key, slice):
            return Trajectory(
                self._positions[key],
                self._orientations[key],
                self._timesteps[key],
                self._parent_frame,
                self._child_frame,
            )
        return Trajectory(
            [self._positions[key]],
            [self._orientations[key]],
            [self._timesteps[key]], # type: ignore
            self._parent_frame,
            self._child_frame,
        )


    def inverse(self) -> Trajectory:
        """Returns the inverse of the trajectory, with the parent and child frames swapped, and the positions and orientations inverted.       
        
        Returns:
            Trajectory: The inverse trajectory.
        """
        inverse = self.as_rigid_unit_quat().inverse()
        return Trajectory(
            inverse.translation,
            inverse.linear,
            self._timesteps,
            self._child_frame,
            self._parent_frame,
        )

    def transform(self, position: torch.Tensor, orientation: torch.Tensor) -> Trajectory:
        """Transforms the trajectory using the given position and orientation. 
        Note this updates the trajectory in place.

        Args:
            position (torch.Tensor): The position to transform the trajectory to. Should be of shape (3,) or (N, 3).
            orientation (torch.Tensor): The orientation to transform the trajectory to. Should be of shape (4,) or (N, 4).

        Returns:
            Trajectory: The transformed trajectory.
        """

        if position.ndim == 1:
            position = position.unsqueeze(0).expand(len(self._positions), -1).to(self._positions.device, self._positions.dtype)
        if orientation.ndim == 1:
            orientation = orientation.unsqueeze(0).expand(len(self._orientations), -1).to(self._orientations.device, self._orientations.dtype)

        T_1 = roma.RigidUnitQuat(translation=position, linear=orientation)
        trajectory = self.as_rigid_unit_quat()
        self._positions, self._orientations = (T_1 @ trajectory).translation, (T_1 @ trajectory).linear
        return self

    def __matmul__(self, other: Trajectory) -> Trajectory:
        """Concatenates two trajectories by applying the transformation of the second trajectory to the first trajectory.

        Args:
            other (Trajectory): The trajectory to concatenate with.

        Returns:
            Trajectory: The concatenated trajectory.
        """

        if not isinstance(other, Trajectory):
            raise ValueError("Trajectory can only be added to another Trajectory object")

        # make sure this is a valid chain
        if self._child_frame != other._parent_frame:
            raise ValueError(
                f"Cannot add trajectories. The child frame of the first trajectory is {self._child_frame} and the parent frame of the second trajectory is {other._parent_frame}"
            )

        # check sampling timesteps of other trajectory
        if self.timesteps.shape != other.timesteps.shape or not self.timesteps.equal(other.timesteps):
            print("Warning: The timesteps of the two trajectories are not equal. The timesteps of the first trajectory will be used and the second trajectory will be resampled with linear interpolation")
            other = other.resample(self._timesteps, interpolation="linear")

        self_as_rigid = self.as_rigid_unit_quat()
        other_as_rigid = other.as_rigid_unit_quat()

        # apply transformation
        concat = (self_as_rigid @ other_as_rigid).normalize()
        return Trajectory(
           concat.translation, concat.linear, self._timesteps, self._parent_frame, other._child_frame
        )


    def as_rigid_unit_quat(self) -> roma.RigidUnitQuat:
        """Converts the trajectory to a rigid unit quaternion representation."""
        return roma.RigidUnitQuat(translation=self._positions, linear=self._orientations)

    def clone(self) -> Trajectory:
        """Creates a deep copy of the trajectory."""
        return Trajectory(
            self._positions.clone(),
            self._orientations.clone(),
            self._timesteps.clone(),
            self._parent_frame,
            self._child_frame,
        )

    def resample(
        self,
        new_timesteps: Sequence[float] | torch.Tensor | float | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        frequency: float | None = None,
        interpolation: str = "linear",
    ) -> Trajectory:
        """Resamples the trajectory based on the given parameters.
        Args:
            new_timesteps (Sequence[float] | torch.Tensor | float | None, optional): The new timesteps to resample the trajectory to. If not provided, the trajectory will be resampled based on the frequency parameter. Defaults to None.
            start_time (float | None, optional): The start time of the resampled trajectory. If not provided, the start time of the original trajectory will be used. Defaults to None.
            end_time (float | None, optional): The end time of the resampled trajectory. If not provided, the end time of the original trajectory will be used. Defaults to None.
            frequency (float | None, optional): The frequency at which to resample the trajectory. Either new_timesteps or frequency must be provided. Defaults to None.
            interpolation (str, optional): The interpolation method to use. Supported methods are 'nearest' and 'linear'. Defaults to "linear".
        Returns:
            Trajectory: The resampled trajectory.
        Raises:
            ValueError: If neither new_timesteps nor frequency is provided.
            ValueError: If the frequency is too high for the given time range.
            ValueError: If the time range is 0.
            ValueError: If the new timesteps are not increasing.
            ValueError: If the interpolation method is not recognized.
        """

        if new_timesteps is None:
            if frequency is None:
                raise ValueError("Either new_timesteps or frequency must be provided")

            if start_time is None:
                start_time = self.start_time
            if end_time is None:
                end_time = self.end_time

            duration = end_time - start_time
            if duration < 1 / frequency:
                raise ValueError(
                    f"Frequency {frequency} is too high for the given time range. The time range is {end_time - start_time} seconds, which is less than 1/{frequency} seconds"
                )

            if duration == 0:
                raise ValueError("The time range must be greater than 0")

            steps = duration * frequency
            if steps % 1 != 0:
                print(
                    f"Warning: Frequency {frequency} is not a divisor of the time range {duration}. End range will be adjusted to match the frequency"
                )

            steps = int(steps)
            new_timesteps = torch.linspace(0, end_time-start_time, int(steps) + 1, dtype = self.timesteps.dtype, device = self.timesteps.device) + start_time

        if not isinstance(new_timesteps, torch.Tensor):
            new_timesteps: torch.Tensor = torch.tensor(new_timesteps)
            # check timesteps increasing

        new_timesteps: torch.Tensor = new_timesteps
        if new_timesteps.diff().min() < 0:
            raise ValueError("The new timesteps must be increasing")

        if interpolation == "nearest":
            # Nearest neighbor interpolation
            # find timsteps that are closest to the new timesteps
            closest_indices = (torch.searchsorted(self._timesteps, new_timesteps, right=False) - 1).clamp(min=0)
            next_closest_indices = (closest_indices + 1).clamp(max=len(self._timesteps) - 1)
            indices = torch.where(
                (self._timesteps[closest_indices] - new_timesteps).abs()
                < (self._timesteps[next_closest_indices] - new_timesteps).abs(),
                closest_indices,
                next_closest_indices,
            )
            new_positions = self._positions[indices]
            new_orientations = self._orientations[indices]
            return Trajectory(new_positions, new_orientations, new_timesteps, self._parent_frame, self._child_frame)

        elif interpolation == "linear":
            closest_indices = (torch.searchsorted(self._timesteps, new_timesteps, right=False) - 1).clamp(min=0)
            lower_dist = (self._timesteps[closest_indices] - new_timesteps).abs().float()
            next_closest_indices = (closest_indices + 1).clamp(max=len(self._timesteps) - 1)
            upper_dist = (self._timesteps[next_closest_indices] - new_timesteps).abs().float()
            total_dist = lower_dist + upper_dist + 1e-6

            interpolated_orientation = unitquat_slerp(
                self._orientations[closest_indices], self._orientations[next_closest_indices], lower_dist / total_dist
            )
            interpolated_position = self._positions[closest_indices] + (
                self._positions[next_closest_indices] - self._positions[closest_indices]
            ) * (lower_dist / total_dist).unsqueeze(-1)

            return Trajectory(
                interpolated_position, interpolated_orientation, new_timesteps, self._parent_frame, self._child_frame
            )

        else:
            raise ValueError(
                f"Interpolation method {interpolation} not recognized. Supported methods are 'nearest' and 'linear'"
            )

    def show(self, fig: go.Figure | None = None, show: bool = True, line_color: str = "blue", show_frames = False, frame_scale = 0.05, trace_kwargs={}, time_as_color:bool = False, colorscale = "viridis") -> go.Figure:
        """
        Visualizes the trajectory in a 3D plot.
        Args:
            fig (go.Figure | None, optional): The figure to add the trajectory to. If None, a new figure will be created. Defaults to None.
            show (bool, optional): Whether to display the figure. Defaults to True.
            line_color (str, optional): The color of the trajectory line. Defaults to "blue".
            show_frames (bool, optional): Whether to show the orientation frames. Defaults to False.
            frame_scale (float, optional): The scale of the orientation frames. Defaults to 0.05.
            trace_kwargs (dict, optional): Additional keyword arguments to pass to the go.Scatter3d traces. Defaults to {}.
            time_as_color (bool, optional): Whether to use time as color. Defaults to False.
            colorscale (str, optional): The colorscale to use for coloring the trajectory. Defaults to "viridis".
        Returns:
            go.Figure: The figure containing the trajectory plot.
        """
        
        legendgroup = self._parent_frame + " to " + self._child_frame

        if fig is None:
            fig = go.Figure()
            # check if figure already contains a trace with the same name

        fig: go.Figure

        cnter = 0
        while any(trace.legendgroup == legendgroup for trace in fig.data):
            cnter += 1
            legendgroup = self._parent_frame + " to " + self._child_frame + f"_{cnter}"

        if time_as_color:
            if isinstance(time_as_color, Callable):
                line_color = time_as_color(self._timesteps)
            else:
                line_color = self._timesteps

        fig.add_trace(
            go.Scatter3d(
                x=self._positions[:, 0],
                y=self._positions[:, 1],
                z=self._positions[:, 2],
                mode="markers+lines",
                line=dict(color=line_color, width=3, colorscale=colorscale),
                marker=dict(size=3, color=line_color, colorscale=colorscale),
                name=f"{self._child_frame}",
                legendgroup=legendgroup,
            ),
            **trace_kwargs
        )


        if show_frames:
            # Add small axes to show orientation
            coordinate_directions = frame_scale * torch.tensor(
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=self._positions.dtype, device=self._positions.device
            )
            # apply rotation to the coordinate directions
            rotated_directions = roma.unitquat_to_rotmat(self._orientations).inverse() @ coordinate_directions.unsqueeze(0).expand(
                len(self._positions), -1, -1
            )
            for idx, (pos, orientation) in enumerate(zip(self._positions, rotated_directions)):
                if isinstance(show_frames, int) and idx % show_frames != 0:
                    continue

                xi, yi, zi = pos
                # Small x-axis (along X direction)
                fig.add_trace(
                    go.Scatter3d(
                        x=[xi, xi + orientation[0, 0]],
                        y=[yi, yi + orientation[0, 1]],
                        z=[zi, zi + orientation[0, 2]],
                        mode="lines",
                        line=dict(color="red", width=2),
                        showlegend=False,
                        legendgroup=legendgroup,
                        name=f"{self._child_frame}",
                    ),
                    **trace_kwargs
                )
                # Small y-axis (along Y direction)
                fig.add_trace(
                    go.Scatter3d(
                        x=[xi, xi + orientation[1, 0]],
                        y=[yi, yi + orientation[1, 1]],
                        z=[zi, zi + orientation[1, 2]],
                        mode="lines",
                        line=dict(color="green", width=2),
                        showlegend=False,
                        legendgroup=legendgroup,
                    ),
                    **trace_kwargs
                )
                # Small z-axis (along Z direction)
                fig.add_trace(
                    go.Scatter3d(
                        x=[xi, xi + orientation[2, 0]],
                        y=[yi, yi + orientation[2, 1]],
                        z=[zi, zi + orientation[2, 2]],
                        mode="lines",
                        line=dict(color="blue", width=2),
                        showlegend=False,
                        legendgroup=legendgroup,
                    ),
                    **trace_kwargs
                )

        # fix aspect ratio
        fig.update_layout(scene=dict(aspectmode="data"))
        if show:
            fig.show()

        return fig

    def temporal_align(self, other: Trajectory, translation = True, orientation: bool = True, max_delay: float = torch.inf, return_infos = False) -> Trajectory:
        """
        Temporally aligns the current trajectory with another trajectory.
        Args:
            other (Trajectory): The trajectory to align with.
            translation (bool, optional): Whether to align the translation. Defaults to True.
            orientation (bool, optional): Whether to align the orientation. Defaults to True.
            max_delay (float, optional): The maximum delay allowed for alignment. Defaults to torch.inf.
            return_infos (bool, optional): Whether to return alignment information. Defaults to False.
        Returns:
            Tuple[Trajectory, Trajectory] or Tuple[Trajectory, Trajectory, Dict[str, Any]]: 
            If return_infos is False, returns a tuple containing the aligned trajectory of the current object and the aligned trajectory of the other object.
            If return_infos is True, returns a tuple containing the aligned trajectory of the current object, the aligned trajectory of the other object, and a dictionary containing alignment information (rotation, translation, delay).
        """
        
        if self.duration < other.duration:
            raise ValueError("The first trajectory must have a longer duration than the second trajectory")

        other_start_ts = other.start_time

        best_error = 1e8
        best_trajectory = None
        best_other = None
        best_r = None
        best_t = None
        best_delay = None

        for start_ts in self._timesteps:
            start_ts = start_ts.item()
            if (start_ts - self.start_time) > max_delay:
                break

            reference_traj = self.slice(start_time=start_ts, end_time=start_ts + other.duration).clone()
            if reference_traj.duration < other.duration - 0.1:
                break

            delay = other_start_ts - reference_traj.start_time
            other_delayed = other.clone()
            other_delayed._timesteps = other_delayed._timesteps - delay
            # resample other trajectory to match the reference trajectory
            other_delayed = other_delayed.resample(reference_traj.timesteps, interpolation="linear")
            # calculate the error after aligning the two trajectories
            other_delayed = other_delayed.spatial_align(reference_traj, translation=translation, orientation=orientation, return_infos=return_infos)
         
            if return_infos:
                other_delayed, r_rotation, t_translation = other_delayed
            error = ((reference_traj.positions - other_delayed.positions)**2).mean()

            if error < best_error:
                best_error = error
                best_trajectory = reference_traj.clone()
                best_other = other_delayed.clone()
                best_delay = delay

                if return_infos:
                    best_r = roma.rotmat_to_unitquat(r_rotation)
                    best_t = t_translation

        if return_infos:
            return best_trajectory, best_other, {"rotation": best_r, "translation": best_t, "delay": best_delay}
        return best_trajectory, best_other

    def spatial_align(self, other: Trajectory, translation:bool = True, orientation:bool = True, return_infos = False) -> Trajectory:
        """Aligns the current trajectory with another trajectory in terms of translation and/or orientation.
        
        Args:
            other (Trajectory): The trajectory to align with.
            translation (bool, optional): Whether to align the translation. Defaults to True.
            orientation (bool, optional): Whether to align the orientation. Defaults to True.
            return_infos (bool, optional): Whether to return additional alignment information. Defaults to False.
        Returns:
            Trajectory: The aligned trajectory.
        Raises:
            ValueError: If the timesteps of the two trajectories are not equal.
        Note:
            - If `translation` is True and `orientation` is False, only the translation of the trajectory will be aligned.
            - If `orientation` is True, both translation and orientation will be aligned.
            - If `return_infos` is True, additional alignment information will be returned.
        """
    

        if self._timesteps.shape != other.timesteps.shape or not self._timesteps.equal(other.timesteps):
            raise ValueError("The timesteps of the two trajectories are not equal. The trajectories must be resampled to have the same timesteps")

        if translation and not orientation:
            print("Warning: Aligning only translation will not change the orientation of the trajectory")
            t_registration = other.avg_position - self.avg_position
            positions = self._positions + t_registration
            R_registration = torch.zeros(len(self._orientations), 4)
            R_registration[:, -1] = 1
            if return_infos:
                return Trajectory(positions, self._orientations, self._timesteps, self._parent_frame, self._child_frame), R_registration, t_registration
            return Trajectory(positions, self._orientations, self._timesteps, self._parent_frame, self._child_frame)

        if orientation:
            R_registration, t_registration = roma.rigid_points_registration(self._positions, other.positions)
            if return_infos:
                return self.clone().transform(t_registration, roma.rotmat_to_unitquat(R_registration)), R_registration, t_registration
            return self.clone().transform(t_registration, roma.rotmat_to_unitquat(R_registration))
        return self

    """
    " Properties of the Trajectory class
    """

    @property
    def avg_position(self):
        return self._positions.mean(dim=0)

    @property
    def avg_orientation(self):
        avg_rotation_mat = roma.special_procrustes(self.as_rigid_unit_quat().to_homogeneous()[..., :3,:3].sum(dim=0))
        return roma.rotmat_to_unitquat(avg_rotation_mat)

    def average(self):
        return Trajectory(self.avg_position.unsqueeze(0), self.avg_orientation.unsqueeze(0), [self._timesteps.mean().item()], self._parent_frame, self._child_frame)

    @property
    def positions(self):
        return self._positions

    @property
    def orientations(self):
        return self._orientations

    @property
    def timesteps(self):
        return self._timesteps

    @property
    def start_time(self):
        return self._timesteps[0].item()

    @property
    def end_time(self):
        return self._timesteps[-1].item()

    @property
    def duration(self):
        return (self._timesteps[-1] - self._timesteps[0]).item()

    @property
    def parent_frame(self):
        return self._parent_frame

    @parent_frame.setter
    def parent_frame(self, value: str):
        self._parent_frame = value
        
    @property
    def child_frame(self):
        return self._child_frame

    # set child frame
    @child_frame.setter
    def child_frame(self, value: str):
        self._child_frame = value

    @property
    def euclidean_length(self):
        return torch.linalg.norm(self._positions[1:] - self._positions[:-1], dim=1).sum().item()

    @property
    def __len__(self):
        return len(self._positions)

    def __repr__(self):
        return f"Trajectory from {self._parent_frame} to {self._child_frame} with {len(self._positions)} points, duration {self.duration}s and euclidean length {self.euclidean_length}m"

    @property
    def so3(self):
        return self.as_rigid_unit_quat().to_homogeneous()[..., :3, :3]