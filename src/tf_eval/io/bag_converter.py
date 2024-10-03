from __future__ import annotations


from pathlib import Path

from rosbags.highlevel import AnyReader
import os
import pandas as pd
from tf_eval.types import TfRecord
import torch
from tf_eval.trajectory import Trajectory
from prettytable import PrettyTable

def tf_to_dict(tf_message):
    return {
        "timestamp": tf_message.header.stamp.sec + tf_message.header.stamp.nanosec * 1e-9,
        "parent_frame": tf_message.header.frame_id.strip("/"),
        "child_frame": tf_message.child_frame_id.strip("/"),
        "translation_x": tf_message.transform.translation.x,
        "translation_y": tf_message.transform.translation.y,
        "translation_z": tf_message.transform.translation.z,
        "rotation_x": tf_message.transform.rotation.x,
        "rotation_y": tf_message.transform.rotation.y,
        "rotation_z": tf_message.transform.rotation.z,
        "rotation_w": tf_message.transform.rotation.w,
    }


def convert_rosbag_to_dfs(bag_file: str, output_dir: str = "data", save= True) -> dict[str, pd.DataFrame]:
    # bag = rosbag.Bag(bag_file)

    tf_messages = []
    # Create reader instance and open for reading.
    with AnyReader([Path(bag_file)]) as reader:
        connections = [x for x in reader.connections if x.topic == '/tf']
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            
            for tf in msg.transforms:
                tf_messages.append(tf_to_dict(tf))

    output_dir = os.path.join(output_dir, os.path.basename(bag_file).replace(".bag", ""))
    os.makedirs(output_dir, exist_ok=True)

    data = {}
    for tf_message in tf_messages:
        child_frame = tf_message["child_frame"]

        if child_frame not in data:
            data[child_frame] = []
        else:
            if data[child_frame][-1]["parent_frame"] != tf_message["parent_frame"]:
                raise ValueError(f"Multiple messages with the same child frame and parent frame found: {tf_message}")

        data[child_frame].append(tf_message)


    table = PrettyTable()
    table.field_names = ["Frame", "Number of messages", "Time range (s)"]
    table.align = "l"

    dfs = {}
    for frame, messages in data.items():
        df = pd.DataFrame(messages)
        frame_name = frame.replace("/", "_")
        if save:
            df.to_csv(os.path.join(output_dir, f"{frame_name}.csv"), index=False)

        dfs[frame] = df
        time_range = (df["timestamp"].max() - df["timestamp"].min())
        table.add_row([frame, len(messages), time_range])
    print(table)

    return dfs



def load_trajectories_from_bag(path_to_rosbag: str) -> dict[str, Trajectory]:
    """Load ground truth trajectories from a rosbag file.
    
    Args:
        path_to_rosbag (str): Path to the rosbag file.
        
    Returns:
        dict: A dictionary of trajectories.
    """
    dfs = convert_rosbag_to_dfs(path_to_rosbag, output_dir="data", save=False)
    trajectories = {}
    for child_frame, df in dfs.items():
        positions = torch.from_numpy(df[["translation_x", "translation_y", "translation_z"]].values).float()
        orientations = torch.from_numpy(df[["rotation_x", "rotation_y", "rotation_z", "rotation_w"]].values).float()
        timesteps = torch.from_numpy(df["timestamp"].values).double()
        parent_frame = "/".join(df["parent_frame"].values[0].split("/")[:-1])
        child_frame = "/".join(df["child_frame"].values[0].split("/")[:-1])
        traj = Trajectory(positions, orientations, timesteps, parent_frame, child_frame).resample(frequency=10)
        trajectories[child_frame] =  traj

    # Uncomment this to convert everything into marker origin frame

    # Convert everything in marker origin frame
    # base_traj = trajectories["vicon/MarkerOrigin/MarkerOrigin"].clone().average().inverse().resample(frequency=10, start_time=trajectories["vicon/MarkerOrigin/MarkerOrigin"].start_time, end_time=trajectories["vicon/MarkerOrigin/MarkerOrigin"].end_time)
    # for key in trajectories.keys():
        # trajectories[key] = base_traj @ trajectories[key]
    return trajectories