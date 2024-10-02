# Trajectory Utils

This package is a collection of utilities for working with trajectories.

## Installation

```bash
pip install -e .
```

## Usage

```bash
python tests/test_bag.py
```

### Evaluate a bag file

```bash
python tests/test_bag.py --bag_file /path/to/bag/file.bag \
    --obj_traj /path/to/object/trajectory.npy --obj_ts /path/to/object/timestamps.npy \
    --glasses_traj /path/to/glasses/trajectory.npy --glasses_ts /path/to/glasses/timestamps.npy \
    --sampling_frequency 30 --time_sync_slack 0.1 --output_folder /path/to/output/folder --output_name output_name
```